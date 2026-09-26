import pytest

pytest.importorskip('torch')
pytest.importorskip('trl')

import torch
from datasets import Dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import GRPOConfig

from bet.parsing import declared_abstention
from bet.rewards import BETRewardConfig, make_trl_reward_functions
from bet.training.trainer import BETGRPOTrainer


@pytest.fixture
def trainer(tmp_path):
    vocab = {token: i for i, token in enumerate(['[PAD]', '[EOS]', '[UNK]'] + list(dict.fromkeys(
        ''.join(chr(i) for i in range(32, 127)) + '\n'
    )))}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Split('', behavior='isolated')
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                       eos_token='[EOS]', unk_token='[UNK]')
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_layer=1, n_head=2, n_embd=16,
                                     n_positions=512, eos_token_id=1, pad_token_id=0))
    args = GRPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to='none',
                      per_device_train_batch_size=2, num_generations=2, max_completion_length=256,
                      max_steps=1, beta=0, loss_type='grpo', epsilon=0.0625, save_strategy='no',
                      gradient_checkpointing=False)
    return BETGRPOTrainer(model=model, args=args, processing_class=tokenizer,
                          reward_funcs=make_trl_reward_functions(BETRewardConfig(max_completion_tokens=256), tokenizer),
                          train_dataset=Dataset.from_list([{'prompt': 'Question?', 'answer': '1'}]))


def scripted_sample(trainer):
    def sample(prompt_ids, limits, *, declare=False):
        tokenizer = trainer.processing_class
        results = []
        for ids, limit in zip(prompt_ids, limits):
            if declare:
                text = '<predict>\nSolvability: 0.00\nBudget: 0.00\n</predict>'
            else:
                prefix = tokenizer.decode(ids)
                text = '<think>Work.</think>\\boxed{1}' if 'Budget: 1.00' in prefix else '<think></think>\\boxed{Unsolvable}'
            generated = tokenizer.encode(text, add_special_tokens=False)
            if not declare:
                generated.append(tokenizer.eos_token_id)
            assert len(generated) <= limit
            results.append(generated)
        return results
    return sample


def test_two_phase_rollout_and_one_training_step(trainer, monkeypatch):
    monkeypatch.setattr(trainer, '_sample', scripted_sample(trainer))
    inputs = [{'prompt': 'Question?', 'answer': '1'}] * 2
    output = trainer._generate_and_score_completions(inputs)
    texts = trainer.processing_class.batch_decode(output['completion_ids'], skip_special_tokens=True)
    assert sum(declared_abstention(text) for text in texts) == 1
    forced = next(i for i, n in enumerate(trainer._bet_prefix_lengths) if n)
    length = trainer._bet_prefix_lengths[forced]
    assert output['completion_mask'][forced, :length].all()
    assert not output['policy_loss_mask'][forced, :length].any()
    assert output['policy_loss_mask'][forced, length:].any()
    before = next(trainer.model.parameters()).detach().clone()
    trainer.train()
    assert trainer.state.global_step == 1
    assert not torch.equal(before, next(trainer.model.parameters()).detach())


def test_override_masks_gradient_without_hiding_prefix(trainer, monkeypatch):
    logps = torch.zeros((2, 4), requires_grad=True)
    def get_logps(model, ids, attention_mask, length):
        assert attention_mask.all()
        return logps, None
    monkeypatch.setattr(trainer, '_get_per_token_logps_and_entropies', get_logps)
    trainer.current_gradient_accumulation_steps = 1
    inputs = dict(prompt_ids=torch.ones((2, 2), dtype=torch.long), prompt_mask=torch.ones((2, 2)),
                  completion_ids=torch.ones((2, 4), dtype=torch.long), completion_mask=torch.ones((2, 4)),
                  policy_loss_mask=torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
                  advantages=torch.ones(2))
    trainer._compute_loss(trainer.model, inputs).backward()
    assert torch.equal(logps.grad[0, :2], torch.zeros(2))
    assert (logps.grad[0, 2:] != 0).all()
    assert (logps.grad[1] != 0).all()


def test_global_reward_profile_sees_success_on_another_rank(trainer, monkeypatch):
    import bet.training.trainer as module
    prompts = ['p', 'p']
    completions = ['<predict>\nSolvability: 0\nBudget: 0\n</predict><think></think>\\boxed{Unsolvable}',
                   '<predict>\nSolvability: 0.5\nBudget: 1\n</predict><think>Work</think>\\boxed{1}']
    gathered = iter([[{'answer': '1'}] * 2, prompts, completions, [[], []]])
    monkeypatch.setattr(module, 'gather_object', lambda _: next(gathered))
    rewards = trainer._calculate_rewards([{'answer': '1'}], prompts[:1], completions[:1], [[]])
    value_column = trainer.reward_func_names.index('bet_value')
    assert rewards[:, value_column].tolist() == pytest.approx([-0.8, 1])


def test_local_backend_respects_completion_limit(trainer):
    prompt = trainer.processing_class.encode('Question?', add_special_tokens=False)
    result = trainer._sample([prompt], [4], declare=True)
    assert 0 < len(result[0]) <= 4


def test_cold_start_adapter_is_merged_and_trainable(trainer, tmp_path):
    peft = pytest.importorskip('peft')
    from bet.training.model_utils import load_policy_model
    base_path = tmp_path / 'base'
    adapter_path = tmp_path / 'adapter'
    trainer.model.save_pretrained(base_path)
    base = GPT2LMHeadModel.from_pretrained(base_path)
    adapter = peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=4, target_modules=['c_attn'],
                                                      task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name, parameter in adapter.named_parameters():
            if 'lora_B' in name:
                parameter.fill_(0.1)
    adapter.eval()
    ids = torch.tensor([[3, 4, 5]])
    expected = adapter(ids).logits.detach()
    adapter.save_pretrained(adapter_path)
    merged = load_policy_model(str(adapter_path))
    merged.eval()
    assert all(parameter.requires_grad for parameter in merged.parameters())
    assert not any('lora_' in name for name, _ in merged.named_parameters())
    torch.testing.assert_close(merged(ids).logits, expected, atol=1e-5, rtol=1e-5)


def test_vllm_phases_preserve_prefixes_and_share_one_length_cap(trainer, monkeypatch):
    from types import SimpleNamespace
    tokenizer = trainer.processing_class
    calls = []
    def generate(prompts, **kwargs):
        calls.append(kwargs)
        declaration = bool(kwargs['generation_kwargs'].get('stop'))
        completions = []
        for prompt in prompts:
            text = '<predict>\nSolvability: 0\nBudget: 0\n</predict>' if declaration else '<think>Work</think>\\boxed{1}'
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not declaration:
                ids.append(tokenizer.eos_token_id)
            completions.append(ids)
        return {'prompt_ids': [tokenizer.encode(p, add_special_tokens=False) for p in prompts],
                'completion_ids': completions}
    trainer.use_vllm = True
    trainer.vllm_client = SimpleNamespace(generate=generate)
    trainer._last_loaded_step = trainer.state.global_step
    _, completions, _, _ = trainer._generate_single_turn(['Question?'] * 2, None)
    assert len(completions) == 2
    assert sum(n > 0 for n in trainer._bet_prefix_lengths) == 1
    assert all(len(ids) <= trainer.max_completion_length for ids in completions)
    assert calls[0]['generation_kwargs']['stop'] == ['</predict>']
    assert calls[0]['generation_kwargs']['include_stop_str_in_output']
    assert all(call['max_tokens'] < trainer.max_completion_length for call in calls[1:])
    assert all(call['n'] == 1 for call in calls)

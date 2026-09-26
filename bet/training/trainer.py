from __future__ import annotations

import random
from collections import defaultdict
from contextlib import nullcontext

import torch
from accelerate.utils import broadcast_object_list, gather_object
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import StoppingCriteria, StoppingCriteriaList
from trl import GRPOTrainer
from trl.models import unwrap_model_for_generation

from ..constants import PREDICT_END
from ..group_constraint import apply_guaranteed_attempt


class PredictEndCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tokenizer.batch_decode(input_ids[:, self.prompt_length:], skip_special_tokens=False)
        return torch.tensor([PREDICT_END in text for text in texts], device=input_ids.device)


class BETGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.loss_type != 'grpo' or self.beta != 0 or self.use_liger_loss:
            raise ValueError('BET requires loss_type=grpo, beta=0, and use_liger_loss=False.')
        if self.use_vllm and (self.vllm_mode != 'server' or self.vllm_importance_sampling_correction):
            raise ValueError('BET requires vLLM server mode with importance sampling correction disabled.')
        self._bet_rng = random.Random(self.args.seed)

    def _sample(self, prompt_ids, limits, *, declare=False):
        tokenizer = self.processing_class
        if self.use_vllm:
            requests = gather_object(list(zip(prompt_ids, limits)))
            payload = [None]
            if self.accelerator.is_main_process:
                results = [[] for _ in requests]
                buckets = defaultdict(list)
                for index, (_, limit) in enumerate(requests):
                    if limit > 0:
                        buckets[limit].append(index)
                for limit, indices in buckets.items():
                    ids = [requests[i][0] for i in indices]
                    generation_kwargs = dict(self.args.generation_kwargs or {})
                    if declare:
                        generation_kwargs.update(stop=[PREDICT_END], include_stop_str_in_output=True)
                    output = self.vllm_client.generate(
                        prompts=tokenizer.batch_decode(ids, skip_special_tokens=False),
                        n=1,
                        max_tokens=limit,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        repetition_penalty=self.repetition_penalty,
                        generation_kwargs=generation_kwargs,
                    )
                    if output['prompt_ids'] != ids:
                        raise ValueError('The vLLM tokenizer changed the rollout prefix token IDs.')
                    for i, completion in zip(indices, output['completion_ids']):
                        results[i] = completion
                payload[0] = results
            broadcast_object_list(payload, from_process=0)
            start = self.accelerator.process_index * len(prompt_ids)
            return payload[0][start:start + len(prompt_ids)]

        results = []
        with (
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            for ids, limit in zip(prompt_ids, limits):
                if limit <= 0:
                    results.append([])
                    continue
                inputs = torch.tensor([ids], device=self.accelerator.device)
                stopping = StoppingCriteriaList([PredictEndCriteria(tokenizer, len(ids))]) if declare else None
                output = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    max_new_tokens=limit,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=0 if self.top_k is None else self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    stopping_criteria=stopping,
                    synced_gpus=False,
                    disable_compile=True,
                )
                results.append(output[0, len(ids):].tolist())
        return results

    def _generate_single_turn(self, prompts, images):
        if images is not None or any(not isinstance(p, str) for p in prompts):
            raise ValueError('BET expects text prompts rendered with the BET chat template.')
        if self.use_vllm and self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step
        tokenizer = self.processing_class
        prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        if self.max_prompt_length is not None:
            prompt_ids = [ids[-self.max_prompt_length:] for ids in prompt_ids]
        declaration_ids = self._sample(prompt_ids, [self.max_completion_length] * len(prompts), declare=True)
        declarations = tokenizer.batch_decode(declaration_ids, skip_special_tokens=False)
        all_prompts = gather_object(prompts)
        all_declarations = gather_object(declarations)
        payload = [None]
        if self.accelerator.is_main_process:
            if len(all_prompts) % self.num_generations:
                raise ValueError('Rollout batch must contain complete K-sized groups.')
            texts, flags = [], []
            for start in range(0, len(all_prompts), self.num_generations):
                end = start + self.num_generations
                if len(set(all_prompts[start:end])) != 1:
                    raise ValueError('Rollouts for each query must be contiguous.')
                group, overridden = apply_guaranteed_attempt(
                    all_declarations[start:end], k=self.num_generations, rng=self._bet_rng,
                )
                texts.extend(group)
                flags.extend(overridden)
            payload[0] = texts, flags
        broadcast_object_list(payload, from_process=0)
        start = self.accelerator.process_index * len(prompts)
        texts, flags = payload[0]
        flags = flags[start:start + len(prompts)]
        for i, forced in enumerate(flags):
            if forced:
                declaration_ids[i] = tokenizer.encode(texts[start + i], add_special_tokens=False)
        self._bet_prefix_lengths = [len(ids) if forced else 0 for ids, forced in zip(declaration_ids, flags)]
        prefixes = [p + d for p, d in zip(prompt_ids, declaration_ids)]
        limits = [max(0, self.max_completion_length - len(d)) for d in declaration_ids]
        for i, ids in enumerate(declaration_ids):
            if ids and ids[-1] == self.eos_token_id:
                limits[i] = 0
        tails = self._sample(prefixes, limits)
        completions = [d + t for d, t in zip(declaration_ids, tails)]
        return prompt_ids, completions, None, {}

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        all_inputs = gather_object(inputs)
        all_prompts = gather_object(prompts)
        all_completions = gather_object(completions)
        all_ids = gather_object(completion_ids_list)
        rewards = torch.zeros(len(all_prompts), len(self.reward_funcs), device=self.accelerator.device)
        for start in range(0, len(all_prompts), self.num_generations):
            end = start + self.num_generations
            keys = [key for key in all_inputs[start] if key not in ['prompt', 'completion', 'completion_ids']]
            metadata = {key: [row[key] for row in all_inputs[start:end]] for key in keys}
            for i, function in enumerate(self.reward_funcs):
                values = function(
                    prompts=all_prompts[start:end], completions=all_completions[start:end],
                    completion_ids=all_ids[start:end], trainer_state=self.state, **metadata,
                )
                rewards[start:end, i] = torch.tensor(values, device=rewards.device)
        return rewards

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        loss_mask = output['completion_mask'].clone()
        for row, length in enumerate(self._bet_prefix_lengths):
            loss_mask[row, :length] = 0
        output['policy_loss_mask'] = loss_mask
        return output

    def _compute_loss(self, model, inputs):
        completion_mask = inputs['completion_mask']
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model,
            torch.cat([inputs['prompt_ids'], inputs['completion_ids']], dim=1),
            torch.cat([inputs['prompt_mask'], completion_mask], dim=1),
            inputs['completion_ids'].size(1),
        )
        old_logps = inputs.get('old_per_token_logps', per_token_logps.detach())
        ratio = torch.exp(per_token_logps - old_logps)
        clipped_ratio = ratio.clamp(1 - self.epsilon_low, 1 + self.epsilon_high)
        advantages = inputs['advantages'].unsqueeze(1)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
        loss = (loss * inputs['policy_loss_mask']).sum(-1) / completion_mask.sum(-1).clamp(min=1)
        return loss.mean() / self.current_gradient_accumulation_steps

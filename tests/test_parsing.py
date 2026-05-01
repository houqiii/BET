from bet.parsing import parse_response, extract_boxed


def test_parse_valid_response():
    text = """<predict>
Solvability: 0.8
Budget: 0.1
</predict>
<think>
Compute directly.
</think>
\\boxed{42}"""
    parsed = parse_response(text)
    assert parsed.format_ok
    assert parsed.solvability_pred == 0.8
    assert parsed.budget == 0.1
    assert parsed.boxed == '42'


def test_extract_nested_boxed():
    assert extract_boxed(r"final \boxed{\frac{1}{2}}") == r"\frac{1}{2}"

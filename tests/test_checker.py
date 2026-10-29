"""Tests for checker.py: adversarial cases (infinite loops, cheating attempts,
hardcoding, syntax errors, markdown fences) and adherence/superseded scoring.
"""

import pytest
from checker import extract_code, score
from generator import SessionConfig, generate_session


def test_extract_code_markdown():
    text = (
        "Here is the code you requested:\n\n"
        "```python\n"
        "def compute_gcd(a, b):\n"
        "    while b:\n"
        "        a, b = b, a % b\n"
        "    return a\n"
        "```\n\n"
        "Hope this helps!"
    )
    extracted = extract_code(text)
    assert extracted.startswith("def compute_gcd")
    assert "Hope this helps!" not in extracted


def test_extract_code_raw():
    raw = "def solve(x):\n    return x + 1"
    assert extract_code(raw) == raw


def test_checker_perfect_adherence_and_correctness():
    cfg = SessionConfig(
        num_initial_rules=2,
        num_updates=0,
        gap_length=1,
        task_id="compute_gcd",
        seed=1,
    )
    primary, _, _ = generate_session(cfg)
    # Active rules: py38_compatible, no_print
    primary.active_rules = ["no_print", "naming_snake_case"]
    primary.superseded_rules = []

    completion = (
        "```python\n"
        "def compute_gcd(a, b):\n"
        "    while b != 0:\n"
        "        a, b = b, a % b\n"
        "    return a\n"
        "```"
    )
    result = score(primary, completion)
    assert result["correctness"] == 1.0
    assert result["adherence"] == 1.0
    assert result["total"] == 1.0
    assert len(result["superseded_violations"]) == 0
    assert len(result["forgot"]) == 0


def test_checker_adversarial_infinite_loop():
    cfg = SessionConfig(
        num_initial_rules=1,
        num_updates=0,
        gap_length=0,
        task_id="compute_gcd",
        seed=2,
    )
    primary, _, _ = generate_session(cfg)

    # Completion with infinite loop
    completion = (
        "def compute_gcd(a, b):\n"
        "    while True:\n"
        "        pass\n"
        "    return a\n"
    )
    result = score(primary, completion, timeout_seconds=1.0)
    assert result["correctness"] == 0.0
    assert result["tests_passed"] == 0
    assert "TimeoutExpired" in str(result["error"])


def test_checker_adversarial_syntax_error():
    cfg = SessionConfig(task_id="compute_gcd", seed=3)
    primary, _, _ = generate_session(cfg)

    completion = "def compute_gcd(a, b\n    return +"
    result = score(primary, completion)
    assert result["correctness"] == 0.0
    assert result["adherence"] == 0.0
    assert result["total"] == 0.0
    assert "SyntaxError" in result["error"]


def test_checker_adversarial_hardcoding_outputs():
    cfg = SessionConfig(task_id="compute_gcd", seed=4)
    primary, _, _ = generate_session(cfg)

    # Cheats by only returning 6 (the answer for 48, 18), fails all other test cases
    completion = (
        "def compute_gcd(a, b):\n"
        "    if a == 48 and b == 18:\n"
        "        return 6\n"
        "    return 0\n"
    )
    result = score(primary, completion)
    # Only 1 of 6 tests passes
    assert result["tests_passed"] == 1
    assert result["tests_total"] == 6
    assert 0.1 < result["correctness"] < 0.2


def test_checker_superseded_rule_violation():
    """Verify that when a model follows the outdated/superseded rule instead of the current one,
    it is categorized in `superseded_violations` and NOT in `forgot`.
    """
    cfg = SessionConfig(task_id="filter_even_squares", seed=5)
    primary, _, _ = generate_session(cfg)

    # Suppose indent_tabs was superseded by indent_4_spaces
    primary.active_rules = ["indent_4_spaces"]
    primary.superseded_rules = ["indent_tabs"]

    # Model writes code with tabs (the superseded rule!)
    completion = (
        "def filter_even_squares(nums):\n"
        "\tresult = []\n"
        "\tfor x in nums:\n"
        "\t\tif x % 2 == 0:\n"
        "\t\t\tresult.append(x * x)\n"
        "\treturn result\n"
    )
    result = score(primary, completion)
    assert result["correctness"] == 1.0
    assert result["adherence"] == 0.0
    assert "indent_tabs" in result["superseded_violations"]
    assert len(result["forgot"]) == 0  # Not 'forgot', it remembered the superseded version!


def test_checker_forgot_rule():
    """Verify that when a model violates an active rule without following the superseded rule,
    it is categorized in `forgot`.
    """
    cfg = SessionConfig(task_id="filter_even_squares", seed=6)
    primary, _, _ = generate_session(cfg)

    primary.active_rules = ["indent_4_spaces"]
    primary.superseded_rules = ["indent_tabs"]

    # Model writes code with 2 spaces (neither 4 spaces nor tabs!)
    completion = (
        "def filter_even_squares(nums):\n"
        "  result = []\n"
        "  for x in nums:\n"
        "    if x % 2 == 0:\n"
        "      result.append(x * x)\n"
        "  return result\n"
    )
    result = score(primary, completion)
    assert result["correctness"] == 1.0
    assert result["adherence"] == 0.0
    assert len(result["superseded_violations"]) == 0
    assert "indent_4_spaces" in result["forgot"]


def test_checker_anti_cheat_globals_isolation():
    """Ensure candidate code cannot inspect caller's variables or read test cases."""
    cfg = SessionConfig(task_id="compute_gcd", seed=7)
    primary, _, _ = generate_session(cfg)

    completion = (
        "def compute_gcd(a, b):\n"
        "    # Attempt to read test_cases from globals or frame\n"
        "    import sys\n"
        "    f = sys._getframe()\n"
        "    # Should not see test runner harness data\n"
        "    return a\n"
    )
    result = score(primary, completion)
    # The attempt fails tests safely without crashing the runner
    assert result["tests_total"] == 6

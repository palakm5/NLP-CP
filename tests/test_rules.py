"""Tests for rules.py: verifies passing and failing examples for every rule,
conflict symmetry, compatibility checks, edge-case handling, and
strict mutual exclusivity for all update pairs.
"""

import pytest
from rules import (
    ALL_RULES,
    RULES,
    are_rules_compatible,
    get_all_conflict_pairs,
    get_rule,
    get_valid_update,
    sample_active_rules,
)

# Test cases: for every rule in ALL_RULES, provide passing and failing code samples.
RULE_TEST_CASES = {
    "indent_tabs": {
        "pass": "def solveTask(x):\n\treturn x + 1\n",
        "fail": "def solveTask(x):\n    return x + 1\n",
    },
    "indent_2_spaces": {
        "pass": "def solveTask(x):\n  if x > 0:\n    return x\n  return 0\n",
        "fail": "def solveTask(x):\n    return x + 1\n",
    },
    "indent_4_spaces": {
        "pass": "def solveTask(x):\n    if x > 0:\n        return x\n    return 0\n",
        "fail": "def solveTask(x):\n  return x + 1\n",
    },
    "naming_camel_case": {
        "pass": "def solveTask(val):\n    return val * 2\n",
        "fail": "def solve_task(val):\n    return val * 2\n",
    },
    "naming_snake_case": {
        "pass": "def solve_task(val):\n    return val * 2\n",
        "fail": "def solveTask(val):\n    return val * 2\n",
    },
    "no_print": {
        "pass": "def solveTask(x):\n    return x * 2\n",
        "fail": "def solveTask(x):\n    print('debug')\n    return x * 2\n",
    },
    "require_print_debug": {
        "pass": "def solveTask(x):\n    print('computing', x)\n    return x * 2\n",
        "fail": "def solveTask(x):\n    return x * 2\n",
    },
    "no_type_hints": {
        "pass": "def solveTask(x, y):\n    return x + y\n",
        "fail": "def solveTask(x: int, y: int) -> int:\n    return x + y\n",
    },
    "require_type_hints": {
        "pass": "def solveTask(x: int, y: str) -> str:\n    return y * x\n",
        "fail": "def solveTask(x, y):\n    return x + y\n",
    },
    "require_stdlib_import": {
        "pass": "import math\ndef solveTask(x):\n    return math.sqrt(x)\n",
        "fail": "def solveTask(x):\n    return x * 2\n",
    },
    "no_imports": {
        "pass": "def solveTask(x):\n    return len(str(x))\n",
        "fail": "import math\ndef solveTask(x):\n    return math.sqrt(x)\n",
    },
    "max_line_length_80": {
        "pass": "def solveTask(x):\n    return x + 1\n",
        "fail": "def solveTask(x):\n    # " + ("a" * 85) + "\n    return x\n",
    },
    "docstrings_required": {
        "pass": 'def solveTask(x):\n    """Compute something."""\n    return x + 1\n',
        "fail": "def solveTask(x):\n    return x + 1\n",
    },
    "no_docstrings": {
        "pass": "def solveTask(x):\n    return x + 1\n",
        "fail": 'def solveTask(x):\n    """Compute something."""\n    return x + 1\n',
    },
    "no_list_comprehensions": {
        "pass": "def solveTask(items):\n    res = []\n    for i in items:\n        res.append(i * 2)\n    return res\n",
        "fail": "def solveTask(items):\n    return [i * 2 for i in items]\n",
    },
    "require_list_comprehensions": {
        "pass": "def solveTask(items):\n    return [i * 2 for i in items]\n",
        "fail": "def solveTask(items):\n    res = []\n    for i in items:\n        res.append(i * 2)\n    return res\n",
    },
    "py38_compatible": {
        "pass": "def solveTask(x):\n    if x == 1:\n        return 'one'\n    return 'other'\n",
        "fail": "def solveTask(x):\n    match x:\n        case 1:\n            return 'one'\n        case _:\n            return 'other'\n",
    },
    "require_pattern_matching": {
        "pass": "def solveTask(x):\n    match x:\n        case 1:\n            return 'one'\n        case _:\n            return 'other'\n",
        "fail": "def solveTask(x):\n    if x == 1:\n        return 'one'\n    return 'other'\n",
    },
}


def test_all_rules_covered_in_test_cases():
    """Ensure every rule registered in ALL_RULES has a test case."""
    rule_ids = {r.id for r in ALL_RULES}
    assert rule_ids == set(RULE_TEST_CASES.keys()), (
        f"Missing tests for: {rule_ids - set(RULE_TEST_CASES.keys())}"
    )


@pytest.mark.parametrize("rule_id", list(RULE_TEST_CASES.keys()))
def test_rule_passing_example(rule_id):
    rule = get_rule(rule_id)
    code = RULE_TEST_CASES[rule_id]["pass"]
    assert rule.check(code) is True, f"Rule '{rule_id}' failed on its passing example:\n{code}"


@pytest.mark.parametrize("rule_id", list(RULE_TEST_CASES.keys()))
def test_rule_failing_example(rule_id):
    rule = get_rule(rule_id)
    code = RULE_TEST_CASES[rule_id]["fail"]
    assert rule.check(code) is False, f"Rule '{rule_id}' passed unexpectedly on its failing example:\n{code}"


def test_all_update_pairs_are_mutually_exclusive():
    """Verify that for EVERY (rule_A, rule_B) pair that can be produced in an update,
    the two rules are strictly mutually exclusive: code satisfying A fails B,
    and code satisfying B fails A.
    """
    pairs = get_all_conflict_pairs()
    assert len(pairs) > 0, "No conflict pairs registered"

    for id_a, id_b in pairs:
        rule_a = get_rule(id_a)
        rule_b = get_rule(id_b)
        code_a = RULE_TEST_CASES[id_a]["pass"]
        code_b = RULE_TEST_CASES[id_b]["pass"]

        assert rule_a.check(code_a) is True
        assert rule_b.check(code_b) is True

        # Mutual exclusivity: code for A must fail B
        assert rule_b.check(code_a) is False, (
            f"Mutual exclusivity violation: code for '{id_a}' passed conflicting rule '{id_b}':\n{code_a}"
        )
        # Mutual exclusivity: code for B must fail A
        assert rule_a.check(code_b) is False, (
            f"Mutual exclusivity violation: code for '{id_b}' passed conflicting rule '{id_a}':\n{code_b}"
        )


def test_rule_syntax_error_handling():
    """Checker should return False without crashing on invalid Python code."""
    invalid_python = "def broken(x\n    return x +++"
    for rule in ALL_RULES:
        result = rule.check(invalid_python)
        assert result is False, f"Rule '{rule.id}' should return False on syntax error"


def test_paraphrases_exist():
    """Every rule must have at least 3 distinct paraphrases."""
    for rule in ALL_RULES:
        assert len(rule.paraphrases) >= 3, f"Rule '{rule.id}' has fewer than 3 paraphrases"
        for p in rule.paraphrases:
            assert isinstance(p, str) and len(p.strip()) > 10


def test_conflict_symmetry_and_validity():
    """If rule A conflicts with B, both must exist in registry and conflict with each other."""
    for rule in ALL_RULES:
        for conflict_id in rule.conflicts_with:
            assert conflict_id in RULES, f"Rule '{rule.id}' references nonexistent conflict '{conflict_id}'"
            target_rule = RULES[conflict_id]
            assert rule.id in target_rule.conflicts_with, (
                f"Conflict asymmetry: '{rule.id}' conflicts with '{conflict_id}', but '{conflict_id}' does not conflict with '{rule.id}'"
            )


def test_are_rules_compatible():
    assert are_rules_compatible(["indent_tabs", "naming_camel_case", "no_print"]) is True
    assert are_rules_compatible(["indent_tabs", "indent_2_spaces"]) is False
    assert are_rules_compatible(["naming_camel_case", "naming_snake_case"]) is False
    assert are_rules_compatible(["no_type_hints", "require_type_hints"]) is False
    assert are_rules_compatible(["require_stdlib_import", "no_imports"]) is False


def test_sample_active_rules():
    for count in [1, 2, 3, 5]:
        sampled = sample_active_rules(count)
        assert len(sampled) == count
        ids = [r.id for r in sampled]
        assert are_rules_compatible(ids) is True


def test_get_valid_update():
    active = ["indent_tabs", "naming_camel_case", "no_print"]
    update = get_valid_update(active)
    assert update is not None
    old_id, new_id = update
    assert old_id in active
    new_active = [r for r in active if r != old_id] + [new_id]
    assert are_rules_compatible(new_active) is True

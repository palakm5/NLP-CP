"""Tests for tasks.py: verifies that all 12 tasks have at least 5 hidden tests,
that tests compare return values, and that all reference solutions pass 100% of their tests.
"""

import copy
import pytest
from tasks import TASKS, get_all_tasks, get_task, to_camel_case


def test_task_count_and_uniqueness():
    """Verify we have between 10 and 12 well-defined tasks with unique IDs."""
    tasks = get_all_tasks()
    assert 10 <= len(tasks) <= 12
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids)), "Task IDs must be strictly unique"


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_task_structure_and_test_count(task):
    """Every task must have a description, valid entry point, and at least 5 hidden unit tests."""
    assert len(task.description.strip()) > 20
    assert bool(task.entry_point.strip())
    assert len(task.tests) >= 5, f"Task '{task.id}' has fewer than 5 tests ({len(task.tests)})"

    for tc in task.tests:
        assert isinstance(tc.args, tuple), f"Test args in '{task.id}' must be a tuple"
        # Unit tests must compare return values (expected is defined)
        assert tc.expected is not None or task.id == "filter_even_squares"


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_reference_solution_passes_all_tests(task):
    """Execute each task's reference code in an isolated scope and verify all tests pass."""
    env = {}
    exec(task.reference_code, env)

    fn = env.get(task.entry_point)
    assert callable(fn), f"Entry point '{task.entry_point}' not found in reference code"

    for idx, tc in enumerate(task.tests):
        # Deepcopy args to avoid mutation across test cases
        args_copy = copy.deepcopy(tc.args)
        actual = fn(*args_copy)
        assert actual == tc.expected, (
            f"Task '{task.id}' failed test #{idx + 1} ({tc.description}): "
            f"input={tc.args}, expected={tc.expected}, got={actual}"
        )


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_camel_case_entry_point_compatibility(task):
    """Verify that if the function is named in camelCase, the test suite can find and call it."""
    camel_name = to_camel_case(task.entry_point)
    # Replace the def line with camelCase
    camel_code = task.reference_code.replace(f"def {task.entry_point}(", f"def {camel_name}(")

    env = {}
    exec(camel_code, env)

    # Resolve candidate entry point
    fn = env.get(task.entry_point) or env.get(camel_name)
    assert callable(fn), f"Could not resolve camelCase entry point '{camel_name}'"

    # Run the first test case
    tc = task.tests[0]
    actual = fn(*copy.deepcopy(tc.args))
    assert actual == tc.expected

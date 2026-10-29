"""Verification and scoring engine for Instruction Memory Playground.

Evaluates LLM completions against:
1. Active coding rules (via AST & regex checks in rules.py)
2. Superseded rules (detecting when LLMs follow outdated instructions)
3. Hidden unit tests in an isolated, sandboxed subprocess with timeout and no network.

Robust against:
- Markdown code fences and surrounding commentary
- Syntax errors and malformed output
- Infinite loops and deep recursion
- Attempts to inspect or cheat the test harness
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from generator import SessionSample
from rules import get_rule
from tasks import Task, get_task, to_camel_case


def extract_code(completion: str) -> str:
    """Extract Python source code from a model completion.

    Handles markdown fences (```python ... ```, ``` ... ```) or returns raw code.
    If multiple code blocks are present, selects the one containing function definitions.
    """
    if not completion or not completion.strip():
        return ""

    # Look for fenced code blocks
    pattern = r"```(?:python|py)?\s*\n([\s\S]*?)```"
    matches = re.findall(pattern, completion, re.IGNORECASE)

    if matches:
        # Prefer the block with a function definition if multiple blocks exist
        for block in matches:
            if "def " in block:
                return block.strip()
        # Otherwise return the longest block
        return max(matches, key=len).strip()

    # If no markdown blocks, return stripped completion
    return completion.strip()


# Subprocess sandbox runner template
SANDBOX_RUNNER_SCRIPT = '''
import sys
import json
import copy

def run_tests():
    data = json.loads(sys.stdin.read())
    code_str = data["code"]
    entry_point = data["entry_point"]
    camel_entry_point = data["camel_entry_point"]
    test_cases = data["tests"]

    # Execute candidate code in an isolated scope
    candidate_globals = {
        "__name__": "__main__",
        "__doc__": None,
    }
    
    try:
        compiled = compile(code_str, "<candidate_code>", "exec")
        exec(compiled, candidate_globals)
    except Exception as e:
        print(json.dumps({
            "passed": 0,
            "total": len(test_cases),
            "error": f"Execution error during code definition: {type(e).__name__}: {str(e)}"
        }))
        return

    # Locate entry point (snake_case or camelCase)
    fn = None
    for name in [entry_point, camel_entry_point]:
        candidate = candidate_globals.get(name)
        if callable(candidate):
            fn = candidate
            break

    if fn is None:
        print(json.dumps({
            "passed": 0,
            "total": len(test_cases),
            "error": f"Entry point not found. Expected callable '{entry_point}' or '{camel_entry_point}'."
        }))
        return

    passed_count = 0
    for idx, tc in enumerate(test_cases):
        args = copy.deepcopy(tc["args"])
        expected = tc["expected"]
        try:
            actual = fn(*args)
            if actual == expected:
                passed_count += 1
        except Exception as e:
            # Individual test failure
            pass

    print(json.dumps({
        "passed": passed_count,
        "total": len(test_cases),
        "error": None
    }))

if __name__ == "__main__":
    run_tests()
'''


def run_unit_tests_sandboxed(
    code: str,
    task: Task,
    timeout_seconds: float = 2.5,
) -> Dict[str, Any]:
    """Execute unit tests for a task against candidate code in an isolated subprocess."""
    tests_payload = [
        {"args": tc.args, "expected": tc.expected}
        for tc in task.tests
    ]
    input_data = {
        "code": code,
        "entry_point": task.entry_point,
        "camel_entry_point": task.camel_entry_point,
        "tests": tests_payload,
    }

    # Restrict environment (no PYTHONPATH to prevent importing repo secrets or tests)
    clean_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONUNBUFFERED": "1",
    }

    try:
        proc = subprocess.run(
            [sys.executable, "-u", "-c", SANDBOX_RUNNER_SCRIPT],
            input=json.dumps(input_data),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=clean_env,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": 0,
            "total": len(task.tests),
            "error": f"TimeoutExpired: Execution exceeded {timeout_seconds}s (infinite loop or hang).",
        }
    except Exception as e:
        return {
            "passed": 0,
            "total": len(task.tests),
            "error": f"SubprocessLaunchError: {str(e)}",
        }

    stdout = proc.stdout.strip()
    if not stdout:
        stderr = proc.stderr.strip()
        return {
            "passed": 0,
            "total": len(task.tests),
            "error": f"Subprocess crashed without output. stderr: {stderr}",
        }

    try:
        # If there's logging/print output before our JSON, take the last line
        json_line = stdout.splitlines()[-1]
        res = json.loads(json_line)
        return res
    except json.JSONDecodeError:
        return {
            "passed": 0,
            "total": len(task.tests),
            "error": f"Failed to parse test runner output: {stdout}",
        }


def score(
    sample: Union[SessionSample, Dict[str, Any]],
    completion: str,
    correctness_weight: float = 0.5,
    adherence_weight: float = 0.5,
    timeout_seconds: float = 2.5,
) -> Dict[str, Any]:
    """Score a model completion against sample rules and hidden tests.

    Returns dict containing:
    - correctness: fraction of unit tests passed (0.0 to 1.0)
    - adherence: fraction of active rules satisfied (0.0 to 1.0)
    - superseded_violations: list of superseded rules that the code followed
    - forgot: active rules violated where no conflicting superseded rule was followed
    - total: weighted overall score
    - tests_passed, tests_total, active_rule_results, error, code
    """
    if isinstance(sample, SessionSample):
        task_id = sample.task_id
        active_rules = list(sample.active_rules)
        superseded_rules = list(sample.superseded_rules)
    else:
        task_id = sample["task_id"]
        active_rules = list(sample.get("active_rules", []))
        superseded_rules = list(sample.get("superseded_rules", []))

    task = get_task(task_id)
    code = extract_code(completion)

    # 1. Syntax check
    try:
        ast.parse(code)
        syntax_valid = True
        syntax_error = None
    except SyntaxError as e:
        syntax_valid = False
        syntax_error = f"SyntaxError at line {e.lineno}: {e.msg}"

    if not syntax_valid or not code:
        err = syntax_error or "Empty completion or no code provided."
        return {
            "correctness": 0.0,
            "adherence": 0.0,
            "superseded_violations": [],
            "forgot": list(active_rules),
            "active_rule_results": {rid: False for rid in active_rules},
            "tests_passed": 0,
            "tests_total": len(task.tests),
            "total": 0.0,
            "error": err,
            "code": code,
        }

    # 2. Evaluate Active Rules
    active_results: Dict[str, bool] = {}
    violated_active: List[str] = []
    for rid in active_rules:
        rule = get_rule(rid)
        passed = rule.check(code)
        active_results[rid] = passed
        if not passed:
            violated_active.append(rid)

    adherence = (
        sum(active_results.values()) / len(active_rules)
        if active_rules
        else 1.0
    )

    # 3. Evaluate Superseded Rules
    superseded_violations: List[str] = []
    for s_id in superseded_rules:
        s_rule = get_rule(s_id)
        if s_rule.check(code):
            superseded_violations.append(s_id)

    # 4. Compute 'Forgot' (active rules violated where NO superseded conflicting rule was followed)
    forgot: List[str] = []
    for act_id in violated_active:
        act_rule = get_rule(act_id)
        # Check if the code followed a superseded rule that conflicts with this active rule
        followed_superseded_conflict = any(
            c_id in superseded_violations for c_id in act_rule.conflicts_with
        )
        if not followed_superseded_conflict:
            forgot.append(act_id)

    # 5. Run Unit Tests in Sandbox
    test_res = run_unit_tests_sandboxed(code, task, timeout_seconds=timeout_seconds)
    tests_passed = test_res.get("passed", 0)
    tests_total = test_res.get("total", len(task.tests))
    correctness = (tests_passed / tests_total) if tests_total > 0 else 0.0

    # 6. Calculate Weighted Total
    total_score = (correctness_weight * correctness) + (adherence_weight * adherence)

    return {
        "correctness": round(correctness, 4),
        "adherence": round(adherence, 4),
        "superseded_violations": superseded_violations,
        "forgot": forgot,
        "active_rule_results": active_results,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "total": round(total_score, 4),
        "error": test_res.get("error"),
        "code": code,
    }

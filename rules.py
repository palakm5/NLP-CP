"""Rule library for Instruction Memory Playground.

Each Rule provides:
- id: unique string identifier
- name: human-readable name
- description: concise explanation of the rule
- paraphrases: list of natural-language phrasings for conversation generation
- conflicts_with: list of rule IDs this rule contradicts/overrides (strictly mutually exclusive)
- check(code: str) -> bool: AST or lexical validator returning True if code satisfies rule
"""

from __future__ import annotations

import ast
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

# Builtin standard library modules
STDLIB_MODULES: Set[str] = set(sys.stdlib_module_names) | {
    "math", "re", "json", "sys", "os", "collections", "itertools", "functools",
    "heapq", "bisect", "typing", "string", "datetime", "time", "random",
    "copy", "statistics", "operator", "enum", "dataclasses", "abc"
}


@dataclass
class Rule:
    id: str
    name: str
    description: str
    paraphrases: List[str]
    conflicts_with: List[str] = field(default_factory=list)
    check_fn: Optional[Callable[[str], bool]] = None

    def check(self, code: str) -> bool:
        """Evaluate if the given Python code adheres to this rule."""
        if not self.check_fn:
            return True
        try:
            # All Python code must be syntactically valid to satisfy rules
            if _safe_parse(code) is None:
                return False
            return bool(self.check_fn(code))
        except Exception:
            return False

    def sample_paraphrase(self, rng: Optional[random.Random] = None) -> str:
        r = rng if rng is not None else random
        return r.choice(self.paraphrases)


# ==============================================================================
# Rule Check Implementations
# ==============================================================================

def _safe_parse(code: str) -> Optional[ast.AST]:
    try:
        return ast.parse(code)
    except Exception:
        return None


def _check_indent_tabs(code: str) -> bool:
    """Indentation must use tab characters '\\t'. No spaces for indentation."""
    lines = code.splitlines()
    has_indented_line = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        leading = line[:len(line) - len(line.lstrip())]
        if leading:
            has_indented_line = True
            # Indent must consist only of tab characters
            if " " in leading or not leading.startswith("\t"):
                return False
    return has_indented_line


def _check_indent_2_spaces(code: str) -> bool:
    """Indentation must use exactly 2 spaces per level. No tabs allowed."""
    if "\t" in code:
        return False
    lines = code.splitlines()
    has_indented_line = False
    first_level_indent_seen = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        leading = line[:len(line) - len(line.lstrip())]
        if leading:
            has_indented_line = True
            if len(leading) % 2 != 0:
                return False
            if not first_level_indent_seen:
                if len(leading) == 2:
                    first_level_indent_seen = True
                else:
                    return False
    return has_indented_line and first_level_indent_seen


def _check_indent_4_spaces(code: str) -> bool:
    """Indentation must use standard 4 spaces per level. No tabs allowed."""
    if "\t" in code:
        return False
    lines = code.splitlines()
    has_indented_line = False
    first_level_indent_seen = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        leading = line[:len(line) - len(line.lstrip())]
        if leading:
            has_indented_line = True
            if len(leading) % 4 != 0:
                return False
            if not first_level_indent_seen:
                if len(leading) == 4:
                    first_level_indent_seen = True
                else:
                    return False
    return has_indented_line and first_level_indent_seen


def _is_camel_case(name: str) -> bool:
    """camelCase: starts with lowercase, no underscores, has at least one uppercase letter."""
    if not name or "_" in name:
        return False
    return bool(re.match(r"^[a-z]+(?:[A-Z][a-zA-Z0-9]*)+$", name))


def _is_snake_case(name: str) -> bool:
    """snake_case: lowercase with at least one underscore, no uppercase letters."""
    if not name or not name.islower():
        return False
    return "_" in name and bool(re.match(r"^[a-z0-9_]+$", name))


def _check_naming_camel_case(code: str) -> bool:
    """All defined functions must be multi-word camelCase (e.g. solveTask)."""
    tree = _safe_parse(code)
    if not tree:
        return False
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return False
    for fn in functions:
        if fn.name.startswith("__") and fn.name.endswith("__"):
            continue
        if not _is_camel_case(fn.name):
            return False
    return True


def _check_naming_snake_case(code: str) -> bool:
    """All defined functions must be snake_case with underscores (e.g. solve_task)."""
    tree = _safe_parse(code)
    if not tree:
        return False
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return False
    for fn in functions:
        if fn.name.startswith("__") and fn.name.endswith("__"):
            continue
        if not _is_snake_case(fn.name):
            return False
    return True


def _check_no_print(code: str) -> bool:
    """Code must not call print() anywhere."""
    tree = _safe_parse(code)
    if not tree:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                return False
    return True


def _check_require_print_debug(code: str) -> bool:
    """Code must contain at least one call to print()."""
    tree = _safe_parse(code)
    if not tree:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                return True
    return False


def _check_no_type_hints(code: str) -> bool:
    """Function arguments and return values must NOT have type annotations."""
    tree = _safe_parse(code)
    if not tree:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                return False
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.annotation is not None:
                    return False
            if node.args.vararg and node.args.vararg.annotation is not None:
                return False
            if node.args.kwarg and node.args.kwarg.annotation is not None:
                return False
        if isinstance(node, ast.AnnAssign):
            return False
    return True


def _check_require_type_hints(code: str) -> bool:
    """All defined functions must have type hints on arguments and return."""
    tree = _safe_parse(code)
    if not tree:
        return False
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return False
    for node in functions:
        if node.name.startswith("__") and node.name.endswith("__"):
            continue
        if node.returns is None:
            return False
        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            if arg.annotation is None:
                return False
    return True


def _check_require_stdlib_import(code: str) -> bool:
    """Code must import at least one standard library module, and no external packages."""
    tree = _safe_parse(code)
    if not tree:
        return False
    has_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            has_import = True
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg not in STDLIB_MODULES:
                    return False
        elif isinstance(node, ast.ImportFrom):
            has_import = True
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg not in STDLIB_MODULES:
                    return False
    return has_import


def _check_no_imports(code: str) -> bool:
    """Code must not contain any import statements."""
    tree = _safe_parse(code)
    if not tree:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False
    return True


def _check_max_line_length_80(code: str) -> bool:
    """Every line in the code must be 80 characters or fewer."""
    for line in code.splitlines():
        if len(line) > 80:
            return False
    return True


def _check_docstrings_required(code: str) -> bool:
    """Every defined function must include a non-empty docstring."""
    tree = _safe_parse(code)
    if not tree:
        return False
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return False
    for fn in functions:
        if fn.name.startswith("__") and fn.name.endswith("__"):
            continue
        doc = ast.get_docstring(fn)
        if not doc or not doc.strip():
            return False
    return True


def _check_no_docstrings(code: str) -> bool:
    """Functions must NOT have docstrings."""
    tree = _safe_parse(code)
    if not tree:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc and doc.strip():
                return False
    return True


def _check_no_list_comprehensions(code: str) -> bool:
    """Code must NOT use any list comprehensions."""
    tree = _safe_parse(code)
    if not tree:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ListComp):
            return False
    return True


def _check_require_list_comprehensions(code: str) -> bool:
    """Code must include at least one list comprehension."""
    tree = _safe_parse(code)
    if not tree:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ListComp):
            return True
    return False


def _check_py38_compatible(code: str) -> bool:
    """Code must be Python 3.8 compatible: no match/case, no X | Y union types."""
    tree = _safe_parse(code)
    if not tree:
        return False
    for node in ast.walk(tree):
        if hasattr(ast, "Match") and isinstance(node, ast.Match):
            return False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns and _has_bitor_union(node.returns):
                return False
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.annotation and _has_bitor_union(arg.annotation):
                    return False
        if isinstance(node, ast.AnnAssign):
            if node.annotation and _has_bitor_union(node.annotation):
                return False
    return True


def _has_bitor_union(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr):
            return True
    return False


def _check_require_pattern_matching(code: str) -> bool:
    """Code must contain a match / case statement."""
    tree = _safe_parse(code)
    if not tree:
        return False
    if not hasattr(ast, "Match"):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Match):
            return True
    return False


# ==============================================================================
# Rule Definitions Library
# All pairs in conflicts_with are strictly mutually exclusive: no valid code can satisfy both.
# ==============================================================================

ALL_RULES: List[Rule] = [
    Rule(
        id="indent_tabs",
        name="Use Tab Indentation",
        description="Indent all code blocks using tab characters (\\t), not spaces.",
        paraphrases=[
            "Indent all code using tabs instead of spaces.",
            "Use tab characters (\\t) for indentation in every code block.",
            "Please make sure your indentation uses tabs, not spaces.",
            "All block indentations must use single or multiple tab characters.",
            "Remember to indent code with literal tabs rather than spaces."
        ],
        conflicts_with=["indent_2_spaces", "indent_4_spaces"],
        check_fn=_check_indent_tabs,
    ),
    Rule(
        id="indent_2_spaces",
        name="Use 2-Space Indentation",
        description="Indent code blocks using exactly 2 spaces per level. No tabs.",
        paraphrases=[
            "Use exactly 2 spaces for each level of indentation.",
            "Indent your code using 2-space indentation, no tabs.",
            "Please use 2 spaces per indentation level throughout the solution.",
            "Format your code with 2-space indentation.",
            "Remember to indent all inner blocks using 2 spaces, without tabs."
        ],
        conflicts_with=["indent_tabs", "indent_4_spaces"],
        check_fn=_check_indent_2_spaces,
    ),
    Rule(
        id="indent_4_spaces",
        name="Use 4-Space Indentation",
        description="Indent code blocks using standard 4 spaces per level. No tabs.",
        paraphrases=[
            "Use standard 4 spaces for each indentation level.",
            "Format code with 4-space indentation, no tabs.",
            "Please ensure you use 4 spaces per indent level across all code.",
            "Indent using 4 spaces per nesting level.",
            "Write the code strictly formatted with 4-space indentation."
        ],
        conflicts_with=["indent_tabs", "indent_2_spaces"],
        check_fn=_check_indent_4_spaces,
    ),
    Rule(
        id="naming_camel_case",
        name="Use camelCase Function Names",
        description="All defined functions must use multi-word camelCase (e.g. solveTask, findResult).",
        paraphrases=[
            "Name all functions in camelCase (e.g. solveTask, findResult).",
            "Use camelCase naming convention for all defined function names.",
            "Function names must be camelCase starting with a lowercase letter, without underscores.",
            "All function definitions should follow camelCase naming style.",
            "Write function identifiers in camelCase rather than snake_case."
        ],
        conflicts_with=["naming_snake_case"],
        check_fn=_check_naming_camel_case,
    ),
    Rule(
        id="naming_snake_case",
        name="Use snake_case Function Names",
        description="All defined functions must use snake_case with underscores (e.g. solve_task).",
        paraphrases=[
            "Name all functions using snake_case (e.g. solve_task, process_data).",
            "Use lower_snake_case for all function names.",
            "Ensure all function names are lowercase with words separated by underscores.",
            "Use standard snake_case naming for every function.",
            "Function names must be lowercase snake_case without uppercase letters."
        ],
        conflicts_with=["naming_camel_case"],
        check_fn=_check_naming_snake_case,
    ),
    Rule(
        id="no_print",
        name="No Print Calls",
        description="Code must not contain any print() statements.",
        paraphrases=[
            "Do not include any print statements in your code.",
            "No print calls or stdout output allowed in your solution.",
            "Make sure there are no print() calls anywhere in the solution.",
            "Avoid using print(); return values directly without console output.",
            "Never print to stdout; return results cleanly."
        ],
        conflicts_with=["require_print_debug"],
        check_fn=_check_no_print,
    ),
    Rule(
        id="require_print_debug",
        name="Require Debug Print",
        description="Code must include at least one print() call.",
        paraphrases=[
            "Include at least one print statement for debugging or logging.",
            "Add a print() statement inside the function body.",
            "Make sure to include a print call somewhere in the code.",
            "Log intermediate progress or state with a print() statement.",
            "Call print() at least once inside the implementation."
        ],
        conflicts_with=["no_print"],
        check_fn=_check_require_print_debug,
    ),
    Rule(
        id="no_type_hints",
        name="No Type Hints",
        description="Do not use type annotations on parameters, variables, or return values.",
        paraphrases=[
            "Do not use any type hints or type annotations.",
            "Omit all type annotations from function arguments and returns.",
            "Write pure dynamic Python without type hints (no : int, -> str, etc.).",
            "No type annotations allowed in function signatures or bodies.",
            "Avoid type hints entirely in parameter declarations and return signatures."
        ],
        conflicts_with=["require_type_hints"],
        check_fn=_check_no_type_hints,
    ),
    Rule(
        id="require_type_hints",
        name="Require Type Hints",
        description="Every function must have type hints on parameters and return value.",
        paraphrases=[
            "Include type hints for all function parameters and the return value.",
            "Annotate every function parameter and return type with explicit type hints.",
            "Use full type annotations (e.g. def foo(x: int) -> int:) on all functions.",
            "All function signatures must have comprehensive type annotations.",
            "Make sure every argument and the return type have type hints."
        ],
        conflicts_with=["no_type_hints"],
        check_fn=_check_require_type_hints,
    ),
    Rule(
        id="require_stdlib_import",
        name="Require Standard Library Import",
        description="Code must import at least one standard library module (no external packages).",
        paraphrases=[
            "Import at least one standard library module (e.g. math, collections, itertools).",
            "Your solution must include a standard library import.",
            "Use standard library modules to assist the solution; at least one import is required.",
            "Make sure to import a helper module from the standard library.",
            "Rely on standard library imports in your implementation."
        ],
        conflicts_with=["no_imports"],
        check_fn=_check_require_stdlib_import,
    ),
    Rule(
        id="no_imports",
        name="No Imports",
        description="Code must not contain any import statements.",
        paraphrases=[
            "Do not import any modules at all (no standard library or external imports).",
            "Write the solution without any import statements whatsoever.",
            "No imports allowed; use built-in functions only.",
            "Avoid all import and from-import statements.",
            "Rely only on built-in Python primitives without importing anything."
        ],
        conflicts_with=["require_stdlib_import"],
        check_fn=_check_no_imports,
    ),
    Rule(
        id="max_line_length_80",
        name="Max Line Length 80",
        description="Every line of code must be 80 characters or fewer.",
        paraphrases=[
            "Keep all line lengths under 80 characters.",
            "Maximum line length must be 80 characters or fewer for all lines.",
            "No line in the code may exceed 80 characters.",
            "Wrap lines so that every line has at most 80 characters.",
            "Ensure strict PEP 8 line length limit: maximum 80 characters per line."
        ],
        conflicts_with=[],
        check_fn=_check_max_line_length_80,
    ),
    Rule(
        id="docstrings_required",
        name="Docstrings Required",
        description="Every defined function must include a docstring.",
        paraphrases=[
            "Every function must have a docstring explaining what it does.",
            "Include a docstring in every function definition.",
            "Add docstrings to all functions.",
            "All functions must contain a descriptive docstring.",
            "Document every function with a docstring at the top of the body."
        ],
        conflicts_with=["no_docstrings"],
        check_fn=_check_docstrings_required,
    ),
    Rule(
        id="no_docstrings",
        name="No Docstrings",
        description="Functions must not have docstrings.",
        paraphrases=[
            "Do not write any docstrings in functions.",
            "Omit docstrings from all functions.",
            "No docstrings allowed inside function bodies.",
            "Do not document functions with docstrings.",
            "Leave out function docstrings completely."
        ],
        conflicts_with=["docstrings_required"],
        check_fn=_check_no_docstrings,
    ),
    Rule(
        id="no_list_comprehensions",
        name="No List Comprehensions",
        description="Code must not use list comprehensions (use loops or built-ins).",
        paraphrases=[
            "Do not use list comprehensions. Use explicit for loops or while loops instead.",
            "Avoid list comprehensions in your code; use normal loops.",
            "No list comprehensions allowed (no [x for x in ...]).",
            "Do not write list comprehensions; use standard iteration constructs.",
            "Forbid list comprehensions; build lists iteratively with loops."
        ],
        conflicts_with=["require_list_comprehensions"],
        check_fn=_check_no_list_comprehensions,
    ),
    Rule(
        id="require_list_comprehensions",
        name="Require List Comprehensions",
        description="Code must include at least one list comprehension.",
        paraphrases=[
            "You must use at least one list comprehension in your code.",
            "Include at least one list comprehension ([x for ...]) in the implementation.",
            "Use list comprehensions to construct lists concisely.",
            "At least one list comprehension must be present in your solution.",
            "Make sure your code uses a list comprehension."
        ],
        conflicts_with=["no_list_comprehensions"],
        check_fn=_check_require_list_comprehensions,
    ),
    Rule(
        id="py38_compatible",
        name="Python 3.8 Compatible",
        description="Code must be compatible with Python 3.8 (no match/case, no X | Y union types).",
        paraphrases=[
            "Code must be compatible with Python 3.8: do not use match/case or union type syntax (X | Y).",
            "Ensure Python 3.8 compatibility: avoid 'match' statements and '|' union syntax.",
            "Write Python 3.8 compatible code with no modern syntax like structural pattern matching.",
            "Ensure code runs on Python 3.8 without newer language constructs like match or pipe unions.",
            "Avoid Python 3.10+ syntax features: use if/elif instead of match/case and Optional[T] instead of T | None."
        ],
        conflicts_with=["require_pattern_matching"],
        check_fn=_check_py38_compatible,
    ),
    Rule(
        id="require_pattern_matching",
        name="Require Pattern Matching",
        description="Code must use Python 3.10+ match/case statement.",
        paraphrases=[
            "Use Python's match/case statement (pattern matching) in your solution.",
            "Incorporate a match ... case construct in the implementation.",
            "Write your conditional logic using a match-case pattern.",
            "Utilize structural pattern matching (match/case) in the function.",
            "Your solution must include a match statement."
        ],
        conflicts_with=["py38_compatible"],
        check_fn=_check_require_pattern_matching,
    ),
]

RULES: Dict[str, Rule] = {rule.id: rule for rule in ALL_RULES}


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_rule(rule_id: str) -> Rule:
    """Retrieve rule by its ID."""
    if rule_id not in RULES:
        raise KeyError(f"Unknown rule ID: '{rule_id}'")
    return RULES[rule_id]


def get_conflicting_rules(rule_id: str) -> List[str]:
    """Return all rule IDs that conflict with the given rule."""
    return list(get_rule(rule_id).conflicts_with)


def get_all_conflict_pairs() -> List[Tuple[str, str]]:
    """Return all directed (rule_A, rule_B) pairs that can be produced in updates."""
    pairs: List[Tuple[str, str]] = []
    for r in ALL_RULES:
        for c in r.conflicts_with:
            pairs.append((r.id, c))
    return pairs


def are_rules_compatible(rule_ids: Sequence[str]) -> bool:
    """Check whether a set of rule IDs contains any mutually conflicting rules."""
    seen: Set[str] = set()
    for rid in rule_ids:
        rule = get_rule(rid)
        for conflict in rule.conflicts_with:
            if conflict in seen:
                return False
        seen.add(rid)
    return True


def sample_active_rules(n: int, rng: Optional[random.Random] = None) -> List[Rule]:
    """Sample `n` mutually compatible rules randomly."""
    if n < 1:
        return []
    r = rng if rng is not None else random
    shuffled = list(ALL_RULES)
    r.shuffle(shuffled)

    selected: List[Rule] = []
    forbidden_ids: Set[str] = set()

    for rule in shuffled:
        if rule.id in forbidden_ids:
            continue
        selected.append(rule)
        forbidden_ids.add(rule.id)
        forbidden_ids.update(rule.conflicts_with)
        if len(selected) == n:
            break

    if len(selected) < n:
        raise ValueError(f"Could not sample {n} mutually compatible rules (max possible: {len(selected)}).")
    return selected


def get_valid_update(active_rule_ids: List[str], rng: Optional[random.Random] = None) -> Optional[Tuple[str, str]]:
    """Select an active rule that has a valid conflict replacement, and return (old_rule_id, new_rule_id).

    Ensures the new_rule_id is strictly mutually exclusive with old_rule_id
    and does not conflict with any OTHER active rules.
    """
    r = rng if rng is not None else random
    candidates: List[Tuple[str, str]] = []

    active_set = set(active_rule_ids)

    for old_id in active_rule_ids:
        old_rule = get_rule(old_id)
        for new_id in old_rule.conflicts_with:
            if new_id in active_set:
                continue
            new_rule = get_rule(new_id)
            # Check if new_rule conflicts with any active rule OTHER THAN old_id
            other_active = active_set - {old_id}
            if any(conflict in other_active for conflict in new_rule.conflicts_with):
                continue
            candidates.append((old_id, new_id))

    if not candidates:
        return None
    return r.choice(candidates)

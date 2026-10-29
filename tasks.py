"""Task pool for Instruction Memory Playground.

12 algorithmic tasks with strict specifications, 5+ hidden unit tests each,
and reference implementations.
Unit tests compare return values only (never stdout).
Several tasks naturally tempt models into violations (e.g., standard library
imports or list comprehensions) to effectively test rule adherence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def to_camel_case(snake_str: str) -> str:
    """Convert snake_case string to camelCase."""
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def to_snake_case(camel_str: str) -> str:
    """Convert camelCase string to snake_case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str).lower()


@dataclass
class TestCase:
    args: Tuple[Any, ...]
    expected: Any
    description: str = ""


@dataclass
class Task:
    id: str
    name: str
    entry_point: str
    description: str
    tests: List[TestCase] = field(default_factory=list)
    reference_code: str = ""

    @property
    def camel_entry_point(self) -> str:
        return to_camel_case(self.entry_point)


# ==============================================================================
# 12 Tasks Definition
# ==============================================================================

TASKS: List[Task] = [
    # --------------------------------------------------------------------------
    # Task 1: filter_even_squares
    # Tempts: List comprehension ([x * x for x in nums if x % 2 == 0])
    # --------------------------------------------------------------------------
    Task(
        id="filter_even_squares",
        name="Filter Even Squares",
        entry_point="filter_even_squares",
        description=(
            "Write a function `filter_even_squares(nums)` that takes a list of integers `nums` "
            "and returns a new list containing the squares of only the even numbers, "
            "preserving their relative order. Return an empty list if `nums` is empty or has no even numbers."
        ),
        tests=[
            TestCase(args=([1, 2, 3, 4, 5, 6],), expected=[4, 16, 36], description="Basic mixed list"),
            TestCase(args=([],), expected=[], description="Empty list"),
            TestCase(args=([1, 3, 5, 7],), expected=[], description="All odd numbers"),
            TestCase(args=([2, -4, 6],), expected=[4, 16, 36], description="Negative even numbers"),
            TestCase(args=([0, 10, -2],), expected=[0, 100, 4], description="Zero and mixed evens"),
            TestCase(args=([11, 22, 33, 44],), expected=[484, 1936], description="Larger integers"),
        ],
        reference_code=(
            "def filter_even_squares(nums):\n"
            "    result = []\n"
            "    for x in nums:\n"
            "        if x % 2 == 0:\n"
            "            result.append(x * x)\n"
            "    return result\n"
        ),
    ),

    # --------------------------------------------------------------------------
    # Task 2: most_frequent_char
    # Tempts: from collections import Counter
    # --------------------------------------------------------------------------
    Task(
        id="most_frequent_char",
        name="Most Frequent Character",
        entry_point="most_frequent_char",
        description=(
            "Write a function `most_frequent_char(s)` that takes a non-empty string `s` and returns "
            "the character that appears most frequently. If there is a tie, return the character "
            "that appears earliest in `s`. The count is case-sensitive."
        ),
        tests=[
            TestCase(args=("banana",), expected="a", description="Clear single majority"),
            TestCase(args=("abcde",), expected="a", description="All unique: first wins"),
            TestCase(args=("aabbcc",), expected="a", description="Three-way tie: first wins"),
            TestCase(args=("bbaacc",), expected="b", description="Tie with different order"),
            TestCase(args=("Hello World!",), expected="l", description="Spaces and punctuation"),
            TestCase(args=("Aa",), expected="A", description="Case sensitivity check"),
        ],
        reference_code=(
            "def most_frequent_char(s):\n"
            "    counts = {}\n"
            "    first_seen = {}\n"
            "    for idx, ch in enumerate(s):\n"
            "        counts[ch] = counts.get(ch, 0) + 1\n"
            "        if ch not in first_seen:\n"
            "            first_seen[ch] = idx\n"
            "    best_ch = s[0]\n"
            "    best_count = -1\n"
            "    for ch, count in counts.items():\n"
            "        if count > best_count:\n"
            "            best_count = count\n"
            "            best_ch = ch\n"
            "        elif count == best_count:\n"
            "            if first_seen[ch] < first_seen[best_ch]:\n"
            "                best_ch = ch\n"
            "    return best_ch\n"
        ),
    ),

    # --------------------------------------------------------------------------
    # Task 3: flatten_matrix
    # Tempts: List comprehension ([x for row in matrix for x in row])
    # --------------------------------------------------------------------------
    Task(
        id="flatten_matrix",
        name="Flatten Matrix",
        entry_point="flatten_matrix",
        description=(
            "Write a function `flatten_matrix(matrix)` that takes a list of lists of integers `matrix` "
            "and returns a 1D list containing all elements in row-major order. If `matrix` is empty "
            "or contains only empty rows, return an empty list."
        ),
        tests=[
            TestCase(args=([[1, 2], [3, 4]],), expected=[1, 2, 3, 4], description="2x2 matrix"),
            TestCase(args=([],), expected=[], description="Empty outer list"),
            TestCase(args=([[], []],), expected=[], description="Outer list with empty sublists"),
            TestCase(args=([[1, 2, 3]],), expected=[1, 2, 3], description="Single row"),
            TestCase(args=([[1], [2], [3]],), expected=[1, 2, 3], description="Single column"),
            TestCase(args=([[1, -2], [3, 0], [5, 6]],), expected=[1, -2, 3, 0, 5, 6], description="3x2 matrix with negatives"),
        ],
        reference_code=(
            "def flatten_matrix(matrix):\n"
            "    flattened = []\n"
            "    for row in matrix:\n"
            "        for item in row:\n"
            "            flattened.append(item)\n"
            "    return flattened\n"
        ),
    ),

    # --------------------------------------------------------------------------
    # Task 4: group_anagrams
    # Tempts: from collections import defaultdict
    # --------------------------------------------------------------------------
    Task(
        id="group_anagrams",
        name="Group Anagrams",
        entry_point="group_anagrams",
        description=(
            "Write a function `group_anagrams(words)` that takes a list of strings `words` and groups "
            "anagrams together into a list of lists. The groups must appear in order of the first word "
            "belonging to that group, and relative order inside each group must be preserved."
        ),
        tests=[
            TestCase(
                args=(["eat", "tea", "tan", "ate", "nat", "bat"],),
                expected=[["eat", "tea", "ate"], ["tan", "nat"], ["bat"]],
                description="Classic anagram set"
            ),
            TestCase(args=([],), expected=[], description="Empty input"),
            TestCase(args=(["a"],), expected=[["a"]], description="Single word"),
            TestCase(
                args=(["cat", "dog", "bird"],),
                expected=[["cat"], ["dog"], ["bird"]],
                description="No anagrams"
            ),
            TestCase(
                args=(["silent", "listen", "enlist"],),
                expected=[["silent", "listen", "enlist"]],
                description="All in one anagram group"
            ),
            TestCase(
                args=(["rat", "tar", "art", "car"],),
                expected=[["rat", "tar", "art"], ["car"]],
                description="Three anagrams plus one separate"
            ),
        ],
        reference_code=(
            "def group_anagrams(words):\n"
            "    groups = {}\n"
            "    group_order = []\n"
            "    for word in words:\n"
            "        key = ''.join(sorted(word))\n"
            "        if key not in groups:\n"
            "            groups[key] = []\n"
            "            group_order.append(key)\n"
            "        groups[key].append(word)\n"
            "    result = []\n"
            "    for k in group_order:\n"
            "        result.append(groups[k])\n"
            "    return result\n"
        ),
    ),

    # --------------------------------------------------------------------------
    # Task 5: compute_gcd
    # Tempts: import math (math.gcd)
    # --------------------------------------------------------------------------
    Task(
        id="compute_gcd",
        name="Compute GCD",
        entry_point="compute_gcd",
        description=(
            "Write a function `compute_gcd(a, b)` that takes two positive integers `a` and `b` "
            "and returns their greatest common divisor (GCD)."
        ),
        tests=[
            TestCase(args=(48, 18), expected=6, description="Standard composite pair"),
            TestCase(args=(101, 103), expected=1, description="Two prime numbers"),
            TestCase(args=(25, 5), expected=5, description="One divides the other"),
            TestCase(args=(7, 7), expected=7, description="Identical numbers"),
            TestCase(args=(1000, 250), expected=250, description="Multiples of 10"),
            TestCase(args=(17, 34), expected=17, description="Double"),
        ],
        reference_code=(
            "def compute_gcd(a, b):\n"
            "    while b != 0:\n"
            "        temp = b\n"
            "        b = a % b\n"
            "        a = temp\n"
            "    return a\n"
        ),
    ),

    # --------------------------------------------------------------------------
    # Task 6: compress_string
    # Tempts: from itertools import groupby
    # --------------------------------------------------------------------------
    Task(
        id="compress_string",
        name="Compress String",
        entry_point="compress_string",
        description=(
            "Write a function `compress_string(text)` that performs run-length encoding. "
            "For each run of consecutive identical characters, append the character and its repeat count. "
            "For example, 'aabcccccaaa' -> 'a2b1c5a3'. Return '' for empty input."
        ),
        tests=[
            TestCase(args=("aabcccccaaa",), expected="a2b1c5a3", description="Standard repeat runs"),
            TestCase(args=("",), expected="", description="Empty string"),
            TestCase(args=("abcdef",), expected="a1b1c1d1e1f1", description="All unique characters"),
            TestCase(args=("aaaa",), expected="a4", description="Single run"),
            TestCase(args=("aA",), expected="a1A1", description="Case-sensitive runs"),
            TestCase(args=("aaabbb",), expected="a3b3", description="Two equal runs"),
        ],
        reference_code=(
            "def compress_string(text):\n"
            "    if not text:\n"
            "        return ''\n"
            "    result = []\n"
            "    current_char = text[0]\n"
            "    count = 1\n"
            "    for i in range(1, len(text)):\n"
            "        if text[i] == current_char:\n"
            "            count += 1\n"
            "        else:\n"
            "            result.append(current_char + str(count))\n"
            "            current_char = text[i]\n"
            "            count = 1\n"
            "    result.append(current_char + str(count))\n"
            "    return ''.join(result)\n"
        ),
    ),

    # --------------------------------------------------------------------------
    # Task 7: clean_word_list
    # Tempts: List comprehension ([w.strip().lower() for w in tokens if w.strip()])
    # --------------------------------------------------------------------------
    Task(
        id="clean_word_list",
        name="Clean Word List",
        entry_point="clean_word_list",
        description=(
            "Write a function `clean_word_list(tokens)` that takes a list of strings `tokens`, "
            "strips leading and trailing whitespace from each, converts each to lowercase, "
            "and discards any strings that become empty. Return the resulting list."
        ),
        tests=[
            TestCase(args=(["  Hello ", "WORLD\t", ""],), expected=["hello", "world"], description="Basic whitespace and case"),
            TestCase(args=([],), expected=[], description="Empty list"),
            TestCase(args=(["   ", "\t\n", " "],), expected=[], description="All whitespace strings"),
            TestCase(args=(["python", "code"],), expected=["python", "code"], description="Already clean strings"),
            TestCase(args=(["  SpAcE  ", "   bar   "],), expected=["space", "bar"], description="Padded words"),
            TestCase(args=(["", "a", "  b  ", "   "],), expected=["a", "b"], description="Interleaved empties"),
        ],
        reference_code=(
            "def clean_word_list(tokens):\n"
            "    cleaned = []\n"
            "    for token in tokens:\n"
            "        processed = token.strip().lower()\n"
            "        if processed:\n"
            "            cleaned.append(processed)\n"
            "    return cleaned\n"
        ),
    ),

    # --------------------------------------------------------------------------
    # Task 8: find_first_missing_positive
    # --------------------------------------------------------------------------
    Task(
        id="find_first_missing_positive",
        name="First Missing Positive",
        entry_point="find_first_missing_positive",
        description=(
            "Write a function `find_first_missing_positive(nums)` that takes a list of integers `nums` "
            "and returns the smallest positive integer (> 0) that does NOT appear in `nums`."
        ),
        tests=[
            TestCase(args=([1, 2, 0],), expected=3, description="Simple consecutive positive"),
            TestCase(args=([3, 4, -1, 1],), expected=2, description="Missing 2 in between"),
            TestCase(args=([7, 8, 9, 11, 12],), expected=1, description="Missing 1 from start"),
            TestCase(args=([],), expected=1, description="Empty input"),
            TestCase(args=([-1, -2, -3],), expected=1, description="All negative numbers"),
            TestCase(args=([1],), expected=2, description="Single 1 present"),
        ],
        reference_code=(
            "def find_first_missing_positive(nums):\n"
            "    num_set = set(nums)\n"
            "    candidate = 1\n"
            "    while candidate in num_set:\n"
            "        candidate += 1\n"
            "    return candidate\n"
        ),
    ),

    # --------------------------------------------------------------------------
    # Task 9: running_averages
    # Tempts: from itertools import accumulate
    # --------------------------------------------------------------------------
    Task(
        id="running_averages",
        name="Running Averages",
        entry_point="running_averages",
        description=(
            "Write a function `running_averages(nums)` that takes a list of numbers `nums` and returns "
            "a list of cumulative averages up to each index, rounded to 2 decimal places. "
            "Return an empty list if `nums` is empty."
        ),
        tests=[
            TestCase(args=([1, 2, 6],), expected=[1.0, 1.5, 3.0], description="Basic integer list"),
            TestCase(args=([],), expected=[], description="Empty list"),
            TestCase(args=([10],), expected=[10.0], description="Single item"),
            TestCase(args=([1, 2, 3, 4, 5],), expected=[1.0, 1.5, 2.0, 2.5, 3.0], description="Ascending integers"),
            TestCase(args=([-2, 2, -4, 4],), expected=[-2.0, 0.0, -1.33, 0.0], description="Alternating signs"),
            TestCase(args=([5, 5, 5],), expected=[5.0, 5.0, 5.0], description="Constant values"),
        ],
        reference_code=(
            "def running_averages(nums):\n"
            "    if not nums:\n"
            "        return []\n"
            "    result = []\n"
            "    total = 0\n"
            "    for i, x in enumerate(nums, 1):\n"
            "        total += x\n"
            "        result.append(round(total / i, 2))\n"
            "    return result\n"
        ),
    ),

    # --------------------------------------------------------------------------
    # Task 10: is_balanced_brackets
    # --------------------------------------------------------------------------
    Task(
        id="is_balanced_brackets",
        name="Balanced Brackets",
        entry_point="is_balanced_brackets",
        description=(
            "Write a function `is_balanced_brackets(text)` that checks whether all bracket characters "
            "('()', '[]', '{}') in the string `text` are balanced and properly matched. "
            "Non-bracket characters should be ignored. Return True if balanced, False otherwise."
        ),
        tests=[
            TestCase(args=("{[()]}",), expected=True, description="Strictly nested valid brackets"),
            TestCase(args=("",), expected=True, description="Empty string is balanced"),
            TestCase(args=("{[(])}",), expected=False, description="Mismatched order"),
            TestCase(args=("a(b[c]d)e",), expected=True, description="Valid brackets with non-bracket chars"),
            TestCase(args=(")",), expected=False, description="Closing bracket without opening"),
            TestCase(args=("((((",), expected=False, description="Unclosed opening brackets"),
        ],
        reference_code=(
            "def is_balanced_brackets(text):\n"
            "    stack = []\n"
            "    matching = {')': '(', ']': '[', '}': '{'}\n"
            "    for ch in text:\n"
            "        if ch in '([{':\n"
            "            stack.append(ch)\n"
            "        elif ch in matching:\n"
            "            if not stack or stack[-1] != matching[ch]:\n"
            "                return False\n"
            "            stack.pop()\n"
            "    return len(stack) == 0\n"
        ),
    ),

    # --------------------------------------------------------------------------
    # Task 11: chunk_list
    # Tempts: List comprehension ([items[i:i + size] for i in range(0, len(items), size)])
    # --------------------------------------------------------------------------
    Task(
        id="chunk_list",
        name="Chunk List",
        entry_point="chunk_list",
        description=(
            "Write a function `chunk_list(items, size)` that splits a list `items` into consecutive chunks "
            "of length `size`. The last chunk may be smaller than `size` if items cannot be divided evenly. "
            "Return an empty list if `items` is empty."
        ),
        tests=[
            TestCase(args=([1, 2, 3, 4, 5], 2), expected=[[1, 2], [3, 4], [5]], description="Uneven division"),
            TestCase(args=([], 3), expected=[], description="Empty list"),
            TestCase(args=([1, 2, 3, 4], 2), expected=[[1, 2], [3, 4]], description="Exact multiple"),
            TestCase(args=([1, 2, 3], 5), expected=[[1, 2, 3]], description="Chunk size larger than list"),
            TestCase(args=([1, 2, 3], 1), expected=[[1], [2], [3]], description="Chunk size 1"),
            TestCase(args=([10, 20, 30, 40, 50, 60, 70], 3), expected=[[10, 20, 30], [40, 50, 60], [70]], description="7 items chunked by 3"),
        ],
        reference_code=(
            "def chunk_list(items, size):\n"
            "    chunks = []\n"
            "    for i in range(0, len(items), size):\n"
            "        chunks.append(items[i:i + size])\n"
            "    return chunks\n"
        ),
    ),

    # --------------------------------------------------------------------------
    # Task 12: sum_of_primes
    # Tempts: from math import isqrt
    # --------------------------------------------------------------------------
    Task(
        id="sum_of_primes",
        name="Sum of Primes",
        entry_point="sum_of_primes",
        description=(
            "Write a function `sum_of_primes(n)` that returns the sum of all prime numbers strictly less than `n`. "
            "If `n <= 2`, return 0."
        ),
        tests=[
            TestCase(args=(10,), expected=17, description="Primes < 10: 2+3+5+7 = 17"),
            TestCase(args=(2,), expected=0, description="No primes strictly less than 2"),
            TestCase(args=(0,), expected=0, description="Zero input"),
            TestCase(args=(20,), expected=77, description="Primes < 20: 17 + 11+13+17+19 = 77"),
            TestCase(args=(30,), expected=129, description="Primes < 30: 77 + 23+29 = 129"),
            TestCase(args=(-5,), expected=0, description="Negative input"),
        ],
        reference_code=(
            "def sum_of_primes(n):\n"
            "    if n <= 2:\n"
            "        return 0\n"
            "    total = 0\n"
            "    for num in range(2, n):\n"
            "        is_prime = True\n"
            "        d = 2\n"
            "        while d * d <= num:\n"
            "            if num % d == 0:\n"
            "                is_prime = False\n"
            "                break\n"
            "            d += 1\n"
            "        if is_prime:\n"
            "            total += num\n"
            "    return total\n"
        ),
    ),
]

TASKS_BY_ID: Dict[str, Task] = {task.id: task for task in TASKS}


def get_task(task_id: str) -> Task:
    """Retrieve task by its ID."""
    if task_id not in TASKS_BY_ID:
        raise KeyError(f"Unknown task ID: '{task_id}'")
    return TASKS_BY_ID[task_id]


def get_all_tasks() -> List[Task]:
    """Return all available tasks."""
    return list(TASKS)

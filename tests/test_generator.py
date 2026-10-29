"""Tests for generator.py: verifies session generation structure,
control variants (zero_gap and inline), message role alternations, and rule consistency.
"""

import pytest
from generator import SessionConfig, generate_session
from rules import are_rules_compatible


def test_generator_structure_and_controls():
    cfg = SessionConfig(num_initial_rules=3, num_updates=1, gap_length=4, task_id="filter_even_squares", seed=42)
    primary, zero_gap, inline = generate_session(cfg)

    # 1. Check variants
    assert primary.variant == "primary"
    assert zero_gap.variant == "zero_gap"
    assert inline.variant == "inline"

    # 2. Check rule tracking
    assert len(primary.active_rules) == 3
    assert len(primary.superseded_rules) == 1
    assert are_rules_compatible(primary.active_rules) is True

    # Same rules across all variants
    assert primary.active_rules == zero_gap.active_rules == inline.active_rules
    assert primary.superseded_rules == zero_gap.superseded_rules == inline.superseded_rules

    # 3. Message counts:
    # Initial rule turn = 2 msgs (user, assistant)
    # 1 update = 2 msgs (user, assistant)
    # Zero gap distractors = 0 msgs -> total = 2 + 2 + 1 = 5 msgs
    assert len(zero_gap.messages) == 5

    # Gap = 4 -> 4 * 2 = 8 distractor msgs -> total = 2 + 2 + 8 + 1 = 13 msgs
    assert len(primary.messages) == 13
    assert len(inline.messages) == 13

    # 4. Final message prompt inspection
    primary_final = primary.messages[-1]["content"]
    inline_final = inline.messages[-1]["content"]

    assert "Remember to strictly follow these rules:" not in primary_final
    assert "Remember to strictly follow these rules:" in inline_final


def test_generator_no_updates_zero_gap():
    cfg = SessionConfig(num_initial_rules=2, num_updates=0, gap_length=0, task_id="compute_gcd", seed=99)
    primary, zero_gap, inline = generate_session(cfg)

    assert len(primary.superseded_rules) == 0
    assert len(primary.active_rules) == 2
    # Initial turn (2) + final turn (1) = 3 messages
    assert len(primary.messages) == 3
    assert len(zero_gap.messages) == 3

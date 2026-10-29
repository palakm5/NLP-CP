"""Session generator for Instruction Memory Playground.

Generates multi-turn benchmark sessions testing instruction retention across:
1. Initial rule announcements
2. Mutually exclusive rule updates (overrides)
3. Distractor gap turns (0 to 40 turns)
4. Final coding task without rule reminders

Also produces two control variants:
- zero_gap: identical rules and updates, but zero distractor turns (D=0)
- inline: identical rules, updates, and distractors, with current rules restated in the final prompt
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rules import (
    ALL_RULES,
    get_rule,
    get_valid_update,
    sample_active_rules,
)
from tasks import TASKS, Task, get_task

# Load distractors bank
DISTRACTORS_PATH = Path(__file__).parent / "data" / "distractors.json"


def load_distractors() -> List[Dict[str, str]]:
    if not DISTRACTORS_PATH.exists():
        raise FileNotFoundError(f"Distractors file not found at {DISTRACTORS_PATH}")
    with open(DISTRACTORS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class SessionConfig:
    num_initial_rules: int = 3
    num_updates: int = 1
    gap_length: int = 10
    task_id: Optional[str] = None
    seed: Optional[int] = None

    def validate(self):
        if not (1 <= self.num_initial_rules <= 5):
            raise ValueError(f"num_initial_rules must be 1..5, got {self.num_initial_rules}")
        if not (0 <= self.num_updates <= 3):
            raise ValueError(f"num_updates must be 0..3, got {self.num_updates}")
        if not (0 <= self.gap_length <= 40):
            raise ValueError(f"gap_length must be 0..40, got {self.gap_length}")


@dataclass
class Message:
    role: str
    content: str


@dataclass
class SessionSample:
    sample_id: str
    variant: str  # "primary", "zero_gap", "inline"
    task_id: str
    active_rules: List[str]
    superseded_rules: List[str]
    messages: List[Dict[str, str]]
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def generate_session(config: SessionConfig) -> Tuple[SessionSample, SessionSample, SessionSample]:
    """Generate a primary benchmark session along with its zero_gap and inline control variants."""
    config.validate()
    rng = random.Random(config.seed)

    # 1. Select Task
    if config.task_id:
        task = get_task(config.task_id)
    else:
        task = rng.choice(TASKS)

    # 2. Sample Initial Compatible Rules
    initial_rule_objs = sample_active_rules(config.num_initial_rules, rng=rng)
    active_rule_ids: List[str] = [r.id for r in initial_rule_objs]
    superseded_rule_ids: List[str] = []

    # Format initial user rule message
    rule_bullets = [f"- {r.sample_paraphrase(rng)}" for r in initial_rule_objs]
    initial_prompt = (
        "Hi! In this session, please make sure you always strictly follow these coding rules "
        "in any code you write:\n" + "\n".join(rule_bullets)
    )
    initial_ack = "Understood! I will strictly follow those coding rules in all code I write."

    base_rule_turns: List[Dict[str, str]] = [
        {"role": "user", "content": initial_prompt},
        {"role": "assistant", "content": initial_ack},
    ]

    # 3. Generate Rule Updates
    updates_applied = 0
    update_turns: List[Dict[str, str]] = []
    for _ in range(config.num_updates):
        update = get_valid_update(active_rule_ids, rng=rng)
        if not update:
            break
        old_id, new_id = update
        old_rule = get_rule(old_id)
        new_rule = get_rule(new_id)

        active_rule_ids.remove(old_id)
        active_rule_ids.append(new_id)
        superseded_rule_ids.append(old_id)
        updates_applied += 1

        update_prompt = (
            f"Update: actually, please note this change to our rules: {new_rule.sample_paraphrase(rng)}. "
            f"This replaces the previous rule regarding {old_rule.name.lower()}."
        )
        update_ack = (
            f"Understood! I have updated my rules. I will follow '{new_rule.name}' instead from now on."
        )
        update_turns.extend([
            {"role": "user", "content": update_prompt},
            {"role": "assistant", "content": update_ack},
        ])

    # 4. Generate Distractor Turns
    distractor_pool = load_distractors()
    # Shuffle or cycle distractors deterministically
    shuffled_distractors = list(distractor_pool)
    rng.shuffle(shuffled_distractors)

    distractor_turns: List[Dict[str, str]] = []
    for i in range(config.gap_length):
        d = shuffled_distractors[i % len(shuffled_distractors)]
        distractor_turns.extend([
            {"role": "user", "content": d["question"]},
            {"role": "assistant", "content": d["answer"]},
        ])

    # 5. Final Task Messages
    clean_task_prompt = (
        f"Here is a coding task. Please implement the solution in Python:\n\n"
        f"{task.description}\n\n"
        f"Provide the complete, self-contained Python code."
    )

    # Inline variant restates all CURRENT active rules in the final prompt
    current_rule_objs = [get_rule(rid) for rid in active_rule_ids]
    inline_bullets = [f"- {r.sample_paraphrase(rng)}" for r in current_rule_objs]
    inline_task_prompt = (
        f"Here is a coding task. Please implement the solution in Python:\n\n"
        f"{task.description}\n\n"
        f"Remember to strictly follow these rules:\n"
        f"{chr(10).join(inline_bullets)}\n\n"
        f"Provide the complete, self-contained Python code."
    )

    base_id = f"{task.id}_r{config.num_initial_rules}_u{updates_applied}_g{config.gap_length}_s{config.seed}"
    cfg_dict = {
        "num_initial_rules": config.num_initial_rules,
        "num_updates": updates_applied,
        "gap_length": config.gap_length,
        "task_id": task.id,
        "seed": config.seed,
    }

    # Primary sample: rules -> updates -> distractors -> final task
    primary_messages = base_rule_turns + update_turns + distractor_turns + [
        {"role": "user", "content": clean_task_prompt}
    ]
    primary_sample = SessionSample(
        sample_id=f"{base_id}_primary",
        variant="primary",
        task_id=task.id,
        active_rules=list(active_rule_ids),
        superseded_rules=list(superseded_rule_ids),
        messages=primary_messages,
        config=cfg_dict,
    )

    # Zero-gap control: rules -> updates -> NO distractors -> final task
    zero_gap_cfg = dict(cfg_dict, gap_length=0)
    zero_gap_messages = base_rule_turns + update_turns + [
        {"role": "user", "content": clean_task_prompt}
    ]
    zero_gap_sample = SessionSample(
        sample_id=f"{base_id}_zero_gap",
        variant="zero_gap",
        task_id=task.id,
        active_rules=list(active_rule_ids),
        superseded_rules=list(superseded_rule_ids),
        messages=zero_gap_messages,
        config=zero_gap_cfg,
    )

    # Inline control: rules -> updates -> distractors -> final task with rules restated
    inline_messages = base_rule_turns + update_turns + distractor_turns + [
        {"role": "user", "content": inline_task_prompt}
    ]
    inline_sample = SessionSample(
        sample_id=f"{base_id}_inline",
        variant="inline",
        task_id=task.id,
        active_rules=list(active_rule_ids),
        superseded_rules=list(superseded_rule_ids),
        messages=inline_messages,
        config=cfg_dict,
    )

    return primary_sample, zero_gap_sample, inline_sample


def generate_benchmark_dataset(
    configs: List[SessionConfig],
    output_path: Optional[str] = None,
) -> List[SessionSample]:
    """Generate benchmark dataset across a grid of configs and optionally save as JSONL."""
    samples: List[SessionSample] = []
    for cfg in configs:
        primary, zero_gap, inline = generate_session(cfg)
        samples.extend([primary, zero_gap, inline])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(s.to_json() + "\n")

    return samples

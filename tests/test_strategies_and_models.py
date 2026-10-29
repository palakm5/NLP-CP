"""Tests for models.py and strategies.py: verifies model caching,
factory dispatching, and all 4 pluggable inference memory strategies.
"""

import pytest
from generator import SessionConfig, generate_session
from models import DiskCache, MockModel, ModelResponse, get_model
from strategies import (
    MemoryStrategy,
    get_all_strategies,
    get_strategy,
    register_strategy,
)


def test_mock_model_generation():
    model = get_model("mock")
    messages = [{"role": "user", "content": "Write compute_gcd"}]
    resp = model.generate(messages)
    assert isinstance(resp, ModelResponse)
    assert resp.cached is False
    assert resp.latency >= 0.0
    assert "compute_gcd" in resp.text


def test_model_disk_caching(tmp_path):
    cache = DiskCache(cache_dir=tmp_path)
    model = MockModel(model_name="mock-cached", enable_cache=True)
    model.cache = cache

    messages = [{"role": "user", "content": "Hello cache"}]
    resp1 = model.generate(messages)
    assert resp1.cached is False

    resp2 = model.generate(messages)
    assert resp2.cached is True
    assert resp2.text == resp1.text


def test_full_context_strategy():
    cfg = SessionConfig(num_initial_rules=2, num_updates=1, gap_length=2, task_id="compute_gcd", seed=10)
    primary, _, _ = generate_session(cfg)

    strategy = get_strategy("full_context")
    prepared = strategy.prepare_messages(primary)

    # Full context must equal sample messages
    assert len(prepared) == len(primary.messages)
    assert prepared == primary.messages


def test_restate_strategy():
    cfg = SessionConfig(num_initial_rules=2, num_updates=1, gap_length=1, task_id="compute_gcd", seed=11)
    primary, _, _ = generate_session(cfg)

    mock_summary = "- Indent with 4 spaces\n- No print statements"
    model = MockModel(custom_response=mock_summary)
    strategy = get_strategy("restate")

    prepared = strategy.prepare_messages(primary, model=model)
    # History + 1 extra assistant summary message + task
    assert len(prepared) == len(primary.messages) + 1
    assert prepared[-2]["role"] == "assistant"
    assert "Active Coding Rules Summary" in prepared[-2]["content"]
    assert mock_summary in prepared[-2]["content"]
    assert prepared[-1] == primary.messages[-1]


def test_notes_strategy():
    cfg = SessionConfig(num_initial_rules=2, num_updates=1, gap_length=1, task_id="compute_gcd", seed=12)
    primary, _, _ = generate_session(cfg)

    mock_notes = "Rule Notes:\n1. camelCase\n2. no print"
    model = MockModel(custom_response=mock_notes)
    strategy = get_strategy("notes")

    prepared = strategy.prepare_messages(primary, model=model)
    # Notes strategy answers from notes ONLY (system message with notes + task message)
    assert len(prepared) == 2
    assert prepared[0]["role"] == "system"
    assert "Strictly adhere to your active notes" in prepared[0]["content"]
    assert mock_notes in prepared[0]["content"]
    assert prepared[1] == primary.messages[-1]


def test_summarize_strategy():
    cfg = SessionConfig(num_initial_rules=2, num_updates=1, gap_length=1, task_id="compute_gcd", seed=13)
    primary, _, _ = generate_session(cfg)

    mock_summary = "Conversation Summary: Only tabs and no imports."
    model = MockModel(custom_response=mock_summary)
    strategy = get_strategy("summarize")

    prepared = strategy.prepare_messages(primary, model=model)
    # Summarize strategy answers from summary ONLY (system message with summary + task message)
    assert len(prepared) == 2
    assert prepared[0]["role"] == "system"
    assert "Conversation Rule Summary" in prepared[0]["content"]
    assert mock_summary in prepared[0]["content"]
    assert prepared[1] == primary.messages[-1]


def test_custom_strategy_pluggability():
    class DummyStrategy(MemoryStrategy):
        @property
        def name(self) -> str:
            return "dummy_last_turn_only"

        @property
        def description(self) -> str:
            return "Passes only the final task turn."

        def prepare_messages(self, sample, model=None):
            return [sample.messages[-1]]

    register_strategy(DummyStrategy())
    retrieved = get_strategy("dummy_last_turn_only")
    assert retrieved.name == "dummy_last_turn_only"
    assert retrieved in get_all_strategies()

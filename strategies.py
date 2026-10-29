"""Inference-time memory strategies for Instruction Memory Playground.

Each strategy implements `prepare_messages(sample, model) -> List[Dict[str, str]]`:
- full_context: pass entire conversation history as-is
- restate: full context + model-generated summary of active rules before the task
- notes: turn-by-turn running notes scratchpad; final task answered from notes ONLY
- summarize: full conversation condensed into an instruction summary; task answered from summary ONLY

Pluggable architecture allowing new strategies to be registered at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from generator import SessionSample
from models import MockModel, ModelBackend


def _get_history_and_task(sample: Union[SessionSample, Dict[str, Any]]) -> tuple[List[Dict[str, str]], Dict[str, str]]:
    messages = sample.messages if isinstance(sample, SessionSample) else sample["messages"]
    history = messages[:-1]
    task_message = messages[-1]
    return list(history), dict(task_message)


class MemoryStrategy(ABC):
    """Abstract base class for inference memory strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier of the strategy."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of how this strategy manages context."""
        pass

    @abstractmethod
    def prepare_messages(
        self,
        sample: Union[SessionSample, Dict[str, Any]],
        model: Optional[ModelBackend] = None,
    ) -> List[Dict[str, str]]:
        """Return the final list of messages to pass to the model for the coding task."""
        pass


class FullContextStrategy(MemoryStrategy):
    """Passes the complete multi-turn conversation as-is."""

    @property
    def name(self) -> str:
        return "full_context"

    @property
    def description(self) -> str:
        return "Standard baseline: supplies the entire conversation history unchanged."

    def prepare_messages(
        self,
        sample: Union[SessionSample, Dict[str, Any]],
        model: Optional[ModelBackend] = None,
    ) -> List[Dict[str, str]]:
        messages = sample.messages if isinstance(sample, SessionSample) else sample["messages"]
        return list(messages)


class RestateStrategy(MemoryStrategy):
    """Generates an intermediate model summary of current rules from history, appended before the task."""

    @property
    def name(self) -> str:
        return "restate"

    @property
    def description(self) -> str:
        return "Full history plus a model-generated summary of active rules appended right before the task."

    def prepare_messages(
        self,
        sample: Union[SessionSample, Dict[str, Any]],
        model: Optional[ModelBackend] = None,
    ) -> List[Dict[str, str]]:
        history, task_message = _get_history_and_task(sample)
        backend = model or MockModel()

        # Prompt the model to extract and resolve active rules from history
        extraction_query = (
            "Review the preceding conversation carefully. List all CURRENT active coding rules "
            "that must be followed, taking into account any rule updates or overrides that replaced "
            "earlier instructions. Output only the active rules as a concise bulleted list."
        )
        extraction_messages = history + [{"role": "user", "content": extraction_query}]
        rule_summary_resp = backend.generate(extraction_messages, temperature=0.0)
        rule_summary = rule_summary_resp.text.strip()

        # Return full context with the extracted rule summary appended before the final task
        return history + [
            {"role": "assistant", "content": f"Current Active Coding Rules Summary:\n{rule_summary}"},
            task_message,
        ]


class NotesStrategy(MemoryStrategy):
    """Turn-by-turn running notes scratchpad. Final task answered from notes ONLY."""

    @property
    def name(self) -> str:
        return "notes"

    @property
    def description(self) -> str:
        return "Maintains a running notes scratchpad turn-by-turn; answers the final task from notes ONLY."

    def prepare_messages(
        self,
        sample: Union[SessionSample, Dict[str, Any]],
        model: Optional[ModelBackend] = None,
    ) -> List[Dict[str, str]]:
        history, task_message = _get_history_and_task(sample)
        backend = model or MockModel()

        running_notes = "No rules noted yet."

        # Incrementally update notes on each user turn
        for msg in history:
            if msg["role"] == "user":
                update_prompt = (
                    f"You maintain a running scratchpad of coding rules and requirements.\n\n"
                    f"CURRENT SCRATCHPAD NOTES:\n{running_notes}\n\n"
                    f"NEW USER MESSAGE:\n{msg['content']}\n\n"
                    f"Update your scratchpad notes to reflect any new rules, changes, or overrides. "
                    f"If a rule has been updated or overridden, replace the old rule with the new one. "
                    f"If the message is an unrelated question, leave the rule notes intact. "
                    f"Provide only the updated scratchpad notes:"
                )
                resp = backend.generate([{"role": "user", "content": update_prompt}], temperature=0.0)
                if resp.text.strip():
                    running_notes = resp.text.strip()

        # The final task is answered from the notes ONLY
        return [
            {
                "role": "system",
                "content": f"You are an expert Python assistant. Strictly adhere to your active notes:\n\n{running_notes}",
            },
            task_message,
        ]


class SummarizeStrategy(MemoryStrategy):
    """Summarizes entire conversation; final task answered from summary ONLY."""

    @property
    def name(self) -> str:
        return "summarize"

    @property
    def description(self) -> str:
        return "Summarizes all instructions and constraints; answers the task from the summary ONLY."

    def prepare_messages(
        self,
        sample: Union[SessionSample, Dict[str, Any]],
        model: Optional[ModelBackend] = None,
    ) -> List[Dict[str, str]]:
        history, task_message = _get_history_and_task(sample)
        backend = model or MockModel()

        summary_query = (
            "Summarize all coding rules, guidelines, and constraints established throughout this "
            "conversation. Ensure that any superseded rules are omitted and only current active rules are listed."
        )
        extraction_messages = history + [{"role": "user", "content": summary_query}]
        resp = backend.generate(extraction_messages, temperature=0.0)
        summary_text = resp.text.strip()

        # Final task answered from summary ONLY
        return [
            {
                "role": "system",
                "content": f"Conversation Rule Summary:\n\n{summary_text}",
            },
            task_message,
        ]


# ==============================================================================
# Strategy Registry
# ==============================================================================

STRATEGIES: Dict[str, MemoryStrategy] = {
    "full_context": FullContextStrategy(),
    "restate": RestateStrategy(),
    "notes": NotesStrategy(),
    "summarize": SummarizeStrategy(),
}


def register_strategy(strategy: MemoryStrategy):
    """Register a new custom memory strategy dynamically."""
    STRATEGIES[strategy.name] = strategy


def get_strategy(name: str) -> MemoryStrategy:
    """Retrieve strategy by identifier."""
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy: '{name}'. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]


def get_all_strategies() -> List[MemoryStrategy]:
    """Return all registered memory strategies."""
    return list(STRATEGIES.values())

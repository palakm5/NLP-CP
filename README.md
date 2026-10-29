# Instruction Memory Playground 🧠

A verifiable benchmark and interactive Streamlit dashboard measuring whether Large Language Models (LLMs) remember and correctly apply **evolving coding rules** given earlier in a multi-turn conversation.

All evaluations are verified 100% deterministically by code (AST analysis, lexical scanners, and sandboxed unit tests) with **zero LLM-as-judge**.

---

## 🎯 How Memory is Measured

1. **Rule Introduction:** Early in a session, the user specifies rules (e.g., *"use tabs"*, *"no print statements"*).
2. **Rule Overrides:** Later in the conversation, some rules are updated/superseded (e.g., *"actually, use 4 spaces instead"*).
3. **Distractor Gap:** Neutral technical/systems questions create a gap of $D$ turns ($0 \le D \le 40$) without mentioning Python syntax.
4. **Final Coding Task:** The model is given a coding task that **never mentions any rules**.

### Rule Outcomes Tracked:
- **Followed Current Rule (Remembered):** Code adheres to the active rule.
- **Followed Superseded Rule (Missed Update):** Code adheres to the old, overridden rule.
- **Followed Neither (Forgot):** Active rule is violated without adhering to the superseded version.

---

## 📁 Repository Layout

```
NLP-CP/
├── rules.py              # Rule library with AST/regex checkers, paraphrases & mutual exclusion
├── tasks.py              # 12 algorithmic tasks with specs, hidden tests & reference solutions
├── generator.py          # Multi-turn conversation generator with zero_gap & inline controls
├── checker.py            # Subprocess sandbox test runner and multi-rule scoring engine
├── models.py             # Unified backends: OpenAI-compatible, Anthropic, Ollama, Mock
├── strategies.py         # Pluggable memory strategies: full_context, restate, notes, summarize
├── run_benchmark.py      # CLI benchmark runner with grid search, resumability, and summary stats
├── app.py                # 5-tab interactive Streamlit dashboard
├── requirements.txt      # Minimal dependencies
├── pyproject.toml        # pytest configuration
├── data/
│   ├── distractors.json  # Neutral computer science Q&A bank
│   └── cached_demos.json # Instant offline examples for zero-API demos
├── results/
│   ├── evaluations.jsonl # Detailed per-sample evaluation logs
│   └── summary.csv       # Aggregate benchmark summary metrics
└── tests/
    ├── test_rules.py     # Passing, failing & mutual exclusivity tests for all rules
    ├── test_tasks.py     # Reference solution validation and hidden test suite checks
    ├── test_generator.py # Session turn structure and control variant verification
    ├── test_checker.py   # Adversarial sandbox tests (infinite loops, cheating, syntax errors)
    └── test_strategies_and_models.py # Model caching, latency tracking, strategy pipelines
```

---

## 🚀 Quickstart & Setup

### 1. Environment Installation
Python 3.10+ is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
Run the comprehensive test suite (100% deterministic, completes in ~1.5 seconds):

```bash
pytest
```

---

## 🧪 Running the Benchmark

Run evaluations across the parameter grid `(models × strategies × configs × variants)`:

```bash
# Run with the fast local mock model
python run_benchmark.py --models mock --limit 12

# Run against an OpenAI-compatible API
export OPENAI_API_KEY="your-api-key"
python run_benchmark.py --models openai/gpt-4o-mini --limit 20 --gaps 0 5 15

# Run against Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-key"
python run_benchmark.py --models anthropic/claude-3-5-haiku-20241022 --limit 20

# Run against a local Ollama instance
python run_benchmark.py --models ollama/llama3.2 --limit 10

# Resume an interrupted benchmark
python run_benchmark.py --resume
```

Results are saved incrementally to `results/evaluations.jsonl` and summarized in `results/summary.csv`.

---

## 🖥️ Launching the Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run app.py
```

### Dashboard Tabs:
1. **🎮 Live Runner:** Run single-session live prompts with custom initial rules, updates, gap lengths, and models. View transcripts with highlighted rule updates and an instant per-rule pass/fail table.
2. **⚖️ Strategy Comparison:** Compare `full_context`, `restate`, `notes`, and `summarize` on the identical session side-by-side.
3. **📊 Benchmark Results:** Interactive Plotly charts displaying memory decay curves ($D = 0 \dots 40$), model rankings, and baselines against `zero_gap` and `inline` controls.
4. **🔍 Error Explorer:** Filter failures by type (*followed superseded*, *forgot*, *test failure*) and drill down into model code and checker diagnostics.
5. **🧪 Try Your Own:** Interactive rule playground where you can select arbitrary rules and test candidate Python code.

---

## ➕ How to Extend

### Adding a New Rule
Open [`rules.py`](file:///Users/palak/Desktop/NLP-CP/rules.py) and add a new `Rule` to `ALL_RULES`:

```python
Rule(
    id="no_semicolons",
    name="No Semicolons",
    description="Do not use semicolons to separate statements.",
    paraphrases=[
        "Do not use semicolons anywhere in your code.",
        "Avoid using semicolons (;) to terminate statements.",
    ],
    conflicts_with=["require_semicolons"],
    check_fn=lambda code: ";" not in code,
)
```

### Adding a New Memory Strategy
Open [`strategies.py`](file:///Users/palak/Desktop/NLP-CP/strategies.py) and subclass `MemoryStrategy`:

```python
class LastKTurnsStrategy(MemoryStrategy):
    @property
    def name(self) -> str:
        return "last_k_turns"

    @property
    def description(self) -> str:
        return "Retains only the most recent K turns."

    def prepare_messages(self, sample, model=None):
        return sample.messages[-3:]

register_strategy(LastKTurnsStrategy())
```

"""Instruction Memory Playground: Live Streamlit Dashboard.

Interactive benchmark and analysis suite for instruction memory in coding agents.
Contains 5 comprehensive tabs:
1. Live Runner: interactive single-session run with highlighted turns and per-rule scorecard
2. Strategy Comparison: side-by-side evaluation of all memory strategies on identical sessions
3. Benchmark Results: interactive Plotly charts with decay curves and control baselines
4. Error Explorer: filter and drill down into forgot vs superseded failures
5. Try Your Own: custom rule playground and test harness
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from checker import extract_code, score
from generator import SessionConfig, generate_session
from models import MockModel, get_model
from rules import ALL_RULES, RULES, get_rule
from strategies import STRATEGIES, get_strategy
from tasks import TASKS, TASKS_BY_ID, Task, get_all_tasks, get_task

# Page config
st.set_page_config(
    page_title="Instruction Memory Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .rule-tag {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .rule-active { background-color: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }
    .rule-superseded { background-color: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .rule-violation { background-color: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }
    .card {
        padding: 1rem 1.2rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        background-color: #F8FAFC;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_evaluations_df() -> pd.DataFrame:
    """Load benchmark evaluations from results/evaluations.jsonl."""
    jsonl_path = Path("results/evaluations.jsonl")
    if not jsonl_path.exists():
        return pd.DataFrame()
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "config" in df.columns:
        df["gap_length"] = df["config"].apply(lambda c: c.get("gap_length", 0) if isinstance(c, dict) else 0)
        df["num_initial_rules"] = df["config"].apply(lambda c: c.get("num_initial_rules", 0) if isinstance(c, dict) else 0)
        df["num_updates"] = df["config"].apply(lambda c: c.get("num_updates", 0) if isinstance(c, dict) else 0)
    df["has_superseded_violation"] = df["superseded_violations"].apply(lambda v: 1 if v and len(v) > 0 else 0)
    df["has_forgot"] = df["forgot"].apply(lambda v: 1 if v and len(v) > 0 else 0)
    return df


@st.cache_data
def load_cached_demos() -> Dict[str, Any]:
    path = Path("data/cached_demos.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Title Header
st.markdown('<div class="main-header">🧠 Instruction Memory Playground</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Benchmark and diagnostic dashboard measuring evolving rule adherence across long agent conversations.</div>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Configuration")
model_options = [
    "mock",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-haiku-20241022",
    "ollama/llama3.2",
]
selected_model_spec = st.sidebar.selectbox("Model Backend", model_options, index=0)
selected_strategy_name = st.sidebar.selectbox("Inference Strategy", list(STRATEGIES.keys()), index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Session Parameters")
task_list = get_all_tasks()
task_options = {t.id: f"{t.name} ({t.id})" for t in task_list}
selected_task_id = st.sidebar.selectbox(
    "Target Task",
    options=list(task_options.keys()),
    format_func=lambda x: task_options[x],
    index=0,
)

num_rules = st.sidebar.slider("Initial Rules", min_value=1, max_value=5, value=2)
num_updates = st.sidebar.slider("Rule Updates", min_value=0, max_value=3, value=1)
gap_length = st.sidebar.slider("Gap Length D (Distractor Turns)", min_value=0, max_value=40, value=3)
seed_val = st.sidebar.number_input("Random Seed", min_value=1, max_value=999999, value=42)
use_cached_fallback = st.sidebar.checkbox("Offline / Demo Mode (instant cached model)", value=True)

# Generate Session
config = SessionConfig(
    num_initial_rules=num_rules,
    num_updates=num_updates,
    gap_length=gap_length,
    task_id=selected_task_id,
    seed=seed_val,
)
primary_sample, zero_gap_sample, inline_sample = generate_session(config)

# Tabs
tabs = st.tabs([
    "🎮 Live Runner",
    "⚖️ Strategy Comparison",
    "📊 Benchmark Results",
    "🔍 Error Explorer",
    "🧪 Try Your Own",
])

# ==============================================================================
# TAB 1: LIVE RUNNER
# ==============================================================================
with tabs[0]:
    st.subheader("Live Session Execution")
    col1, col2 = st.columns([1.1, 1.0])

    with col1:
        st.markdown("#### Conversation Transcript")
        st.caption("Inspect the multi-turn session. Rule declarations and updates are highlighted below.")

        variant_choice = st.radio(
            "Session Variant",
            ["primary", "zero_gap (D=0 control)", "inline (restated rules control)"],
            horizontal=True,
        )
        active_sample = (
            primary_sample
            if "primary" in variant_choice
            else (zero_gap_sample if "zero_gap" in variant_choice else inline_sample)
        )

        # Display badges for active and superseded rules
        st.markdown("**Active Rules:** " + " ".join([f"`{r}`" for r in active_sample.active_rules]))
        if active_sample.superseded_rules:
            st.markdown("**Superseded Rules (Overridden):** " + " ".join([f"`{r}`" for r in active_sample.superseded_rules]))

        # Render conversation box
        with st.container(height=420):
            for idx, msg in enumerate(active_sample.messages, 1):
                role = msg["role"].capitalize()
                content = msg["content"]
                if "always strictly follow these coding rules" in content:
                    st.info(f"**Turn {idx} [{role} - Rule Declaration]**\n\n{content}")
                elif "Update: actually, please note this change" in content:
                    st.warning(f"**Turn {idx} [{role} - Rule Update]**\n\n{content}")
                elif idx == len(active_sample.messages):
                    st.success(f"**Turn {idx} [{role} - Final Task Prompt]**\n\n{content}")
                else:
                    with st.expander(f"Turn {idx} [{role}]: {content[:70]}...", expanded=False):
                        st.markdown(content)

    with col2:
        st.markdown("#### Model Execution & Verification")
        run_btn = st.button("🚀 Run Live Session", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Executing model and sandbox tests..."):
                try:
                    if use_cached_fallback or selected_model_spec == "mock":
                        model = MockModel(model_name=selected_model_spec)
                    else:
                        model = get_model(selected_model_spec)

                    strat = get_strategy(selected_strategy_name)
                    prepared_messages = strat.prepare_messages(active_sample, model=model)
                    resp = model.generate(prepared_messages, temperature=0.0)
                    score_res = score(active_sample, resp.text)
                except Exception as e:
                    st.error(f"Execution failed: {str(e)}")
                    score_res = None

            if score_res:
                # Metric cards
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Overall Score", f"{score_res['total'] * 100:.1f}%")
                m2.metric("Rule Adherence", f"{score_res['adherence'] * 100:.1f}%")
                m3.metric("Correctness", f"{score_res['tests_passed']}/{score_res['tests_total']}")
                m4.metric("Latency", f"{resp.latency:.2f}s")

                st.markdown("##### Per-Rule Pass/Fail Breakdown")
                rule_rows = []
                for r_id in active_sample.active_rules:
                    rule = get_rule(r_id)
                    passed = score_res["active_rule_results"].get(r_id, False)
                    status = "✅ Followed" if passed else "❌ Violated"
                    rule_rows.append({
                        "Rule": rule.name,
                        "Type": "Active",
                        "Status": status,
                        "Details": rule.description,
                    })

                for s_id in active_sample.superseded_rules:
                    s_rule = get_rule(s_id)
                    followed_superseded = s_id in score_res["superseded_violations"]
                    status = "⚠️ Followed (Outdated Violation!)" if followed_superseded else "✅ Ignored (Correct)"
                    rule_rows.append({
                        "Rule": s_rule.name,
                        "Type": "Superseded",
                        "Status": status,
                        "Details": f"Overridden rule: {s_rule.description}",
                    })

                st.dataframe(pd.DataFrame(rule_rows), use_container_width=True, hide_index=True)

                if score_res["superseded_violations"]:
                    st.error(f"**Superseded Rule Violations Detected:** {', '.join(score_res['superseded_violations'])} (The model followed old, overridden instructions!)")
                if score_res["forgot"]:
                    st.warning(f"**Forgotten Rules:** {', '.join(score_res['forgot'])}")

                st.markdown("##### Generated Solution")
                st.code(score_res["code"], language="python")
        else:
            st.info("Click **Run Live Session** to trigger model inference and verify the AST/regex checkers and sandbox tests.")


# ==============================================================================
# TAB 2: STRATEGY COMPARISON
# ==============================================================================
with tabs[1]:
    st.subheader("Inference Strategy Comparison")
    st.caption("Compare how different memory strategies perform on the exact same session configuration.")

    run_comp_btn = st.button("⚖️ Compare All 4 Strategies on Current Session", type="primary")

    if run_comp_btn:
        with st.spinner("Evaluating strategies..."):
            backend = MockModel(model_name="mock") if use_cached_fallback else get_model(selected_model_spec)
            strat_results = []
            for s_name, strat_obj in STRATEGIES.items():
                prep_msgs = strat_obj.prepare_messages(primary_sample, model=backend)
                m_resp = backend.generate(prep_msgs, temperature=0.0)
                eval_res = score(primary_sample, m_resp.text)
                strat_results.append({
                    "Strategy": s_name,
                    "Total Score": eval_res["total"],
                    "Adherence": eval_res["adherence"],
                    "Correctness": eval_res["correctness"],
                    "Superseded Violations": len(eval_res["superseded_violations"]),
                    "Forgot Count": len(eval_res["forgot"]),
                    "Latency (s)": m_resp.latency,
                })

            strat_df = pd.DataFrame(strat_results)
            st.dataframe(strat_df, use_container_width=True, hide_index=True)

            # Bar chart
            fig = px.bar(
                strat_df,
                x="Strategy",
                y=["Adherence", "Correctness", "Total Score"],
                barmode="group",
                title="Performance Metrics Across Memory Strategies",
                color_discrete_sequence=["#3B82F6", "#10B981", "#8B5CF6"],
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click the button above to run this session across `full_context`, `restate`, `notes`, and `summarize`.")


# ==============================================================================
# TAB 3: BENCHMARK RESULTS
# ==============================================================================
with tabs[2]:
    st.subheader("Benchmark Results & Analysis")
    df_evals = load_evaluations_df()

    if df_evals.empty:
        st.warning("No evaluation results found in `results/evaluations.jsonl`. Run `run_benchmark.py` first.")
    else:
        st.caption(f"Loaded {len(df_evals)} benchmark evaluation records from `results/`.")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            # Metric by model
            model_summary = df_evals.groupby("model")[["adherence", "correctness", "total"]].mean().reset_index()
            fig_model = px.bar(
                model_summary,
                x="model",
                y=["adherence", "correctness", "total"],
                barmode="group",
                title="Overall Performance by Model",
            )
            st.plotly_chart(fig_model, use_container_width=True)

        with col_b2:
            # Memory decay curves by gap length D
            primary_df = df_evals[df_evals["variant"] == "primary"]
            decay_df = primary_df.groupby(["gap_length", "strategy"])["adherence"].mean().reset_index()

            # Add zero_gap and inline control baselines
            zg_val = df_evals[df_evals["variant"] == "zero_gap"]["adherence"].mean()
            inline_val = df_evals[df_evals["variant"] == "inline"]["adherence"].mean()

            fig_decay = px.line(
                decay_df,
                x="gap_length",
                y="adherence",
                color="strategy",
                markers=True,
                title="Instruction Memory Decay Curve (Adherence vs Gap Length D)",
                labels={"gap_length": "Distractor Gap Turns (D)", "adherence": "Rule Adherence"},
            )
            fig_decay.add_hline(y=zg_val, line_dash="dash", line_color="green", annotation_text="Zero Gap Baseline")
            fig_decay.add_hline(y=inline_val, line_dash="dot", line_color="purple", annotation_text="Inline Restated Baseline")
            st.plotly_chart(fig_decay, use_container_width=True)

        col_b3, col_b4 = st.columns(2)
        with col_b3:
            # Adherence by number of updates
            upd_df = df_evals.groupby(["num_updates", "strategy"])["adherence"].mean().reset_index()
            fig_upd = px.bar(
                upd_df,
                x="num_updates",
                y="adherence",
                color="strategy",
                barmode="group",
                title="Rule Adherence vs Number of Rule Updates",
            )
            st.plotly_chart(fig_upd, use_container_width=True)

        with col_b4:
            # Superseded violation rate vs number of updates
            sup_df = df_evals.groupby("num_updates")["has_superseded_violation"].mean().reset_index()
            fig_sup = px.line(
                sup_df,
                x="num_updates",
                y="has_superseded_violation",
                markers=True,
                title="Superseded Rule Violation Rate vs Updates",
                labels={"has_superseded_violation": "Superseded Violation Rate"},
            )
            st.plotly_chart(fig_sup, use_container_width=True)


# ==============================================================================
# TAB 4: ERROR EXPLORER
# ==============================================================================
with tabs[3]:
    st.subheader("Error Explorer")
    st.caption("Inspect and diagnose model failures: distinguish between forgetting rules and following superseded ones.")

    df_evals = load_evaluations_df()
    if df_evals.empty:
        st.warning("No evaluations found. Run benchmark to populate error records.")
    else:
        error_type = st.radio(
            "Filter Failures By Type",
            ["All Imperfect Samples", "Followed Superseded Rule (Missed Update)", "Forgot Rule", "Unit Test Failure"],
            horizontal=True,
        )

        if error_type == "Followed Superseded Rule (Missed Update)":
            filtered = df_evals[df_evals["has_superseded_violation"] == 1]
        elif error_type == "Forgot Rule":
            filtered = df_evals[df_evals["has_forgot"] == 1]
        elif error_type == "Unit Test Failure":
            filtered = df_evals[df_evals["correctness"] < 1.0]
        else:
            filtered = df_evals[df_evals["total"] < 1.0]

        st.markdown(f"**Found {len(filtered)} matching evaluation samples.**")

        if not filtered.empty:
            sample_row = filtered.iloc[0]
            selected_idx = st.selectbox(
                "Select Failure Instance to Inspect",
                range(min(len(filtered), 50)),
                format_func=lambda i: f"#{i+1}: {filtered.iloc[i]['sample_id']} | adh: {filtered.iloc[i]['adherence']} | corr: {filtered.iloc[i]['correctness']}",
            )
            selected_record = filtered.iloc[selected_idx]

            col_e1, col_e2 = st.columns([1, 1])
            with col_e1:
                st.markdown("##### Failure Diagnosis")
                st.write(f"**Model:** `{selected_record['model']}`")
                st.write(f"**Strategy:** `{selected_record['strategy']}` | **Variant:** `{selected_record['variant']}`")
                st.write(f"**Active Rules:** {selected_record['active_rules']}")
                st.write(f"**Superseded Rules:** {selected_record['superseded_rules']}")
                st.error(f"**Superseded Violations:** {selected_record['superseded_violations']}")
                st.warning(f"**Forgot Rules:** {selected_record['forgot']}")
                st.write(f"**Unit Tests Passed:** {selected_record['tests_passed']} / {selected_record['tests_total']}")
                if selected_record.get("error"):
                    st.code(selected_record["error"])

            with col_e2:
                st.markdown("##### Candidate Code Output")
                st.code(extract_code(selected_record["completion"]), language="python")


# ==============================================================================
# TAB 5: TRY YOUR OWN
# ==============================================================================
with tabs[4]:
    st.subheader("Custom Rule & Task Playground")
    st.caption("Design arbitrary rule sets and test code snippets against the automatic verification engine.")

    col_t1, col_t2 = st.columns([1, 1])

    with col_t1:
        custom_task_id = st.selectbox(
            "Select Task Spec",
            list(TASKS_BY_ID.keys()),
            format_func=lambda x: TASKS_BY_ID[x].name,
            key="custom_task_select",
        )
        custom_task = get_task(custom_task_id)
        st.info(f"**Task Description:**\n\n{custom_task.description}")

        rule_choices = [r.id for r in ALL_RULES]
        selected_custom_rules = st.multiselect(
            "Select Active Rules",
            options=rule_choices,
            default=["naming_snake_case", "no_print"],
        )

        selected_custom_superseded = st.multiselect(
            "Select Superseded Rules (to test override violations)",
            options=[r for r in rule_choices if r not in selected_custom_rules],
            default=[],
        )

    with col_t2:
        code_input = st.text_area(
            "Candidate Python Code",
            value=custom_task.reference_code,
            height=300,
        )

        test_now = st.button("🔍 Check Code Adherence & Tests", type="primary")

        if test_now:
            custom_sample = {
                "task_id": custom_task_id,
                "active_rules": selected_custom_rules,
                "superseded_rules": selected_custom_superseded,
            }
            score_out = score(custom_sample, code_input)

            st.markdown("#### Verification Results")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Total Score", f"{score_out['total'] * 100:.1f}%")
            sc2.metric("Rule Adherence", f"{score_out['adherence'] * 100:.1f}%")
            sc3.metric("Tests Passed", f"{score_out['tests_passed']}/{score_out['tests_total']}")

            rule_table = []
            for r_id in selected_custom_rules:
                r_obj = get_rule(r_id)
                passed = score_out["active_rule_results"].get(r_id, False)
                rule_table.append({
                    "Rule": r_obj.name,
                    "Status": "✅ Pass" if passed else "❌ Fail",
                    "Details": r_obj.description,
                })
            for s_id in selected_custom_superseded:
                s_obj = get_rule(s_id)
                followed = s_id in score_out["superseded_violations"]
                rule_table.append({
                    "Rule": f"[Superseded] {s_obj.name}",
                    "Status": "⚠️ VIOLATION (Followed Old Rule)" if followed else "✅ Correct (Ignored)",
                    "Details": s_obj.description,
                })

            st.dataframe(pd.DataFrame(rule_table), use_container_width=True, hide_index=True)

            if score_out.get("error"):
                st.error(f"Error: {score_out['error']}")

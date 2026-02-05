"""
Validation Panel Component
UI for 7-Phase validation pipeline
"""
import streamlit as st
import requests
from typing import Dict, List, Optional
import pandas as pd

# API Base URL
API_BASE = "http://localhost:8000"

# Phase descriptions
PHASE_DESCRIPTIONS = {
    "structural": "Checks strategy structure completeness",
    "logical": "Validates logical consistency of rules",
    "backtest": "Runs historical backtest simulation",
    "monte_carlo": "Monte Carlo statistical analysis",
    "walk_forward": "Walk-forward optimization test",
    "out_of_sample": "Out-of-sample performance validation",
    "robustness": "Parameter sensitivity and robustness checks"
}


def render_validation_panel():
    """Render the validation panel"""
    st.header("7-Phase Validation")

    # Initialize session state
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = {}
    if 'selected_strategy_id' not in st.session_state:
        st.session_state.selected_strategy_id = None

    # Strategy Selection
    st.subheader("Strategy Selection")

    col1, col2 = st.columns([3, 1])

    with col1:
        strategy_id = st.text_input(
            "Strategy ID",
            value=st.session_state.selected_strategy_id or "",
            placeholder="Enter strategy ID or select from Top 5"
        )

    with col2:
        if st.button("Load from Top 5"):
            _show_top5_selector()

    # Phase Selection
    st.subheader("Validation Phases")

    phases = st.multiselect(
        "Select phases to run",
        options=[
            "structural",
            "logical",
            "backtest",
            "monte_carlo",
            "walk_forward",
            "out_of_sample",
            "robustness"
        ],
        default=["structural", "logical", "backtest"],
        help="Select phases to include in validation"
    )

    # Phase descriptions
    with st.expander("Phase Descriptions"):
        for phase, desc in PHASE_DESCRIPTIONS.items():
            st.write(f"**{phase.replace('_', ' ').title()}**: {desc}")

    # Run Validation
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run Validation", type="primary", disabled=not strategy_id):
            _run_validation(strategy_id, phases)

    with col2:
        if st.button("Run All Phases", disabled=not strategy_id):
            _run_validation(strategy_id, None)  # None = all phases

    # Results Display
    if st.session_state.validation_results:
        _display_validation_results()


def _show_top5_selector():
    """Show Top 5 strategies for selection"""
    try:
        response = requests.get(f"{API_BASE}/top5/list", timeout=10)
        if response.status_code == 200:
            data = response.json()
            strategies = data.get('strategies', [])

            if strategies:
                st.write("**Select from Top 5:**")
                for s in strategies[:10]:  # Show max 10
                    sid = s.get('id') or s.get('strategy_id')
                    stype = s.get('strategy_type', 'N/A')
                    if st.button(f"{sid} ({stype})", key=f"top5_select_{sid}"):
                        st.session_state.selected_strategy_id = sid
                        st.rerun()
            else:
                st.info("No Top 5 strategies available")
        else:
            st.error("Failed to load Top 5")

    except Exception as e:
        st.error(f"Error: {str(e)}")


def _run_validation(strategy_id: str, phases: Optional[List[str]]):
    """Run validation on a strategy"""
    try:
        with st.spinner("Running validation..."):
            payload = {"strategy_id": strategy_id}
            if phases:
                payload["phases"] = phases

            response = requests.post(
                f"{API_BASE}/validation/validate",
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state.validation_results = data

                if data.get('overall_passed'):
                    st.success(
                        f"Validation PASSED! Score: {data['overall_score']:.2%} "
                        f"({data['phases_completed']} phases in {data['duration_ms']:.0f}ms)"
                    )
                else:
                    st.warning(
                        f"Validation FAILED. Score: {data['overall_score']:.2%} "
                        f"({data['phases_completed']} phases)"
                    )
            elif response.status_code == 404:
                st.error(f"Strategy not found: {strategy_id}")
            else:
                st.error(f"Validation failed: {response.text}")

    except requests.exceptions.Timeout:
        st.error("Validation timed out")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def _display_validation_results():
    """Display validation results"""
    results = st.session_state.validation_results

    st.subheader("Validation Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        passed = results.get('overall_passed', False)
        st.metric(
            "Status",
            "PASSED" if passed else "FAILED",
            delta=None
        )

    with col2:
        st.metric("Overall Score", f"{results.get('overall_score', 0):.2%}")

    with col3:
        st.metric("Phases Completed", results.get('phases_completed', 0))

    with col4:
        st.metric("Duration", f"{results.get('duration_ms', 0):.0f}ms")

    # Phase Results
    st.subheader("Phase Results")

    phase_results = results.get('phase_results', [])

    if phase_results:
        # Create DataFrame for display
        df_data = []
        for pr in phase_results:
            df_data.append({
                'Phase': pr.get('phase', 'N/A').replace('_', ' ').title(),
                'Status': 'PASS' if pr.get('passed') else 'FAIL',
                'Score': f"{pr.get('score', 0):.2%}",
                'Duration': f"{pr.get('duration_ms', 0):.0f}ms",
                'Details': pr.get('details', '')
            })

        df = pd.DataFrame(df_data)

        # Style the dataframe
        def style_status(val):
            if val == 'PASS':
                return 'background-color: #d4edda'
            elif val == 'FAIL':
                return 'background-color: #f8d7da'
            return ''

        styled_df = df.style.applymap(style_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)

        # Detailed metrics per phase
        st.subheader("Detailed Metrics")

        for pr in phase_results:
            phase_name = pr.get('phase', 'N/A').replace('_', ' ').title()
            with st.expander(f"{phase_name} - {'PASS' if pr.get('passed') else 'FAIL'}"):
                st.write(f"**Score:** {pr.get('score', 0):.2%}")
                st.write(f"**Details:** {pr.get('details', 'N/A')}")

                metrics = pr.get('metrics', {})
                if metrics:
                    st.write("**Metrics:**")
                    col1, col2 = st.columns(2)
                    items = list(metrics.items())
                    mid = len(items) // 2

                    with col1:
                        for k, v in items[:mid + 1]:
                            if isinstance(v, float):
                                st.write(f"- {k}: {v:.4f}")
                            else:
                                st.write(f"- {k}: {v}")

                    with col2:
                        for k, v in items[mid + 1:]:
                            if isinstance(v, float):
                                st.write(f"- {k}: {v:.4f}")
                            else:
                                st.write(f"- {k}: {v}")
    else:
        st.info("No phase results available")


def render_batch_validation():
    """Render batch validation interface"""
    st.subheader("Batch Validation")

    # Get Top 5 strategies
    try:
        response = requests.get(f"{API_BASE}/top5/list", timeout=10)
        if response.status_code == 200:
            data = response.json()
            strategies = data.get('strategies', [])

            if not strategies:
                st.info("No strategies in Top 5 for batch validation")
                return

            # Select strategies
            strategy_ids = [s.get('id') or s.get('strategy_id') for s in strategies]
            selected = st.multiselect(
                "Select strategies for batch validation",
                options=strategy_ids,
                default=strategy_ids[:5]
            )

            # Options
            stop_on_failure = st.checkbox(
                "Stop on first failure",
                value=False,
                help="Stop validating a strategy when it fails a phase"
            )

            if st.button("Run Batch Validation", type="primary", disabled=not selected):
                _run_batch_validation(selected, stop_on_failure)

    except Exception as e:
        st.error(f"Error loading strategies: {str(e)}")


def _run_batch_validation(strategy_ids: List[str], stop_on_failure: bool):
    """Run batch validation"""
    try:
        with st.spinner(f"Validating {len(strategy_ids)} strategies..."):
            response = requests.post(
                f"{API_BASE}/validation/batch",
                json={
                    "strategy_ids": strategy_ids,
                    "stop_on_failure": stop_on_failure
                },
                timeout=300
            )

            if response.status_code == 200:
                data = response.json()

                st.success(
                    f"Batch validation complete: "
                    f"{data['passed']}/{data['validated']} passed"
                )

                # Display results
                results = data.get('results', [])
                df_data = []
                for r in results:
                    df_data.append({
                        'Strategy': r.get('strategy_id', 'N/A'),
                        'Passed': 'YES' if r.get('passed') else 'NO',
                        'Score': f"{r.get('score', 0):.2%}" if 'score' in r else 'N/A',
                        'Error': r.get('error', '')
                    })

                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.error(f"Batch validation failed: {response.text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")

"""
Top 5 Panel Component
UI for Top 5 strategy management with accumulation support
"""
import streamlit as st
import requests
from typing import Dict, List
import pandas as pd
from datetime import datetime

# API Base URL
API_BASE = "http://localhost:8000"


def render_top5_panel():
    """Render the Top 5 management panel"""
    st.header("Top 5 Strategies")

    # Initialize session state
    if 'top5_strategies' not in st.session_state:
        st.session_state.top5_strategies = []
    if 'top5_batch_filter' not in st.session_state:
        st.session_state.top5_batch_filter = None

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Current Top 5", "Batch History", "Statistics"])

    with tab1:
        _render_current_top5()

    with tab2:
        _render_batch_history()

    with tab3:
        _render_statistics()


def _render_current_top5():
    """Render current Top 5 strategies"""
    # Load strategies
    if st.button("Refresh", key="refresh_top5"):
        _load_top5_strategies()

    # Auto-load on first render
    if not st.session_state.top5_strategies:
        _load_top5_strategies()

    strategies = st.session_state.top5_strategies

    if not strategies:
        st.info("No strategies in Top 5. Generate and save strategies to see them here.")
        return

    # Filter by batch
    batches = list(set(s.get('batch_id', 'unknown') for s in strategies))
    batches.insert(0, "All Batches")

    selected_batch = st.selectbox(
        "Filter by Batch",
        options=batches,
        index=0
    )

    # Filter strategies
    if selected_batch != "All Batches":
        filtered = [s for s in strategies if s.get('batch_id') == selected_batch]
    else:
        filtered = strategies

    st.write(f"**Showing {len(filtered)} strategies**")

    # Display strategies
    for i, strategy in enumerate(filtered):
        _render_strategy_card(strategy, i)

    # Bulk actions
    st.subheader("Bulk Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Validate All Top 5", type="secondary"):
            _validate_all_top5(filtered)

    with col2:
        if st.button("Clear All Top 5", type="secondary"):
            if st.session_state.get('confirm_clear', False):
                _clear_all_top5()
                st.session_state.confirm_clear = False
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm clearing all Top 5 strategies")


def _render_strategy_card(strategy: Dict, index: int):
    """Render a strategy card"""
    sid = strategy.get('id') or strategy.get('strategy_id', f'strategy_{index}')
    stype = strategy.get('strategy_type', 'N/A')
    template = strategy.get('template_id', 'N/A')
    batch_id = strategy.get('batch_id', 'N/A')

    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

        with col1:
            st.write(f"**{sid}**")
            st.caption(f"Type: {stype}")

        with col2:
            st.write(f"Template: {template}")
            st.caption(f"Batch: {batch_id[:20]}...")

        with col3:
            # Indicators summary
            indicators = strategy.get('indicators', [])
            if indicators:
                ind_names = [ind.get('name', '') for ind in indicators[:3] if isinstance(ind, dict)]
                st.write(f"Indicators: {', '.join(ind_names)}")

            # Risk summary
            risk = strategy.get('risk_config', {})
            if risk:
                st.caption(f"SL: {risk.get('atr_sl_mult', 'N/A')} ATR | TP: {risk.get('atr_tp_mult', 'N/A')} ATR")

        with col4:
            if st.button("Validate", key=f"validate_{sid}"):
                _promote_to_validation(sid)

            if st.button("Details", key=f"details_{sid}"):
                st.session_state.show_detail = sid

        st.divider()

    # Show detail modal
    if st.session_state.get('show_detail') == sid:
        with st.expander(f"Strategy Details: {sid}", expanded=True):
            _render_strategy_detail(strategy)
            if st.button("Close", key=f"close_{sid}"):
                st.session_state.show_detail = None


def _render_strategy_detail(strategy: Dict):
    """Render detailed strategy view"""
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Entry Logic**")
        entry = strategy.get('entry_logic', {})
        st.write(f"Logic Type: {entry.get('logic_type', 'AND')}")
        for cond in entry.get('conditions', []):
            if isinstance(cond, dict):
                st.write(f"- {cond.get('indicator', 'N/A')} {cond.get('operator', '')} {cond.get('compare_to', '')}")

    with col2:
        st.write("**Exit Logic**")
        exit_logic = strategy.get('exit_logic', {})
        st.write(f"Logic Type: {exit_logic.get('logic_type', 'OR')}")
        for cond in exit_logic.get('conditions', []):
            if isinstance(cond, dict):
                st.write(f"- {cond.get('indicator', 'N/A')} {cond.get('operator', '')} {cond.get('compare_to', '')}")

    st.write("**Risk Configuration**")
    risk = strategy.get('risk_config', {})
    cols = st.columns(4)
    metrics = [
        ("SL (ATR)", risk.get('atr_sl_mult', 'N/A')),
        ("TP (ATR)", risk.get('atr_tp_mult', 'N/A')),
        ("Risk/Trade", f"{risk.get('risk_per_trade', 0) * 100:.2f}%"),
        ("Position Sizing", risk.get('position_sizing_method', 'N/A'))
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)

    # Raw JSON
    with st.expander("Raw JSON"):
        st.json(strategy)


def _render_batch_history():
    """Render batch history"""
    st.subheader("Batch History")

    try:
        response = requests.get(f"{API_BASE}/top5/history", timeout=10)
        if response.status_code == 200:
            data = response.json()
            batches = data.get('batches', [])

            if not batches:
                st.info("No batch history available")
                return

            st.write(f"**Total Batches: {data.get('total_batches', 0)}**")

            # Display as table
            df_data = []
            for batch in batches:
                df_data.append({
                    'Batch ID': batch.get('batch_id', 'N/A'),
                    'Created': batch.get('created_at', 'N/A'),
                    'Count': batch.get('strategy_count', 0)
                })

            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)

            # Select batch for details
            selected = st.selectbox(
                "Select batch for details",
                options=[b.get('batch_id') for b in batches],
                index=0
            )

            if st.button("Load Batch"):
                st.session_state.top5_batch_filter = selected
                _load_top5_strategies(batch_id=selected)
                st.rerun()

    except Exception as e:
        st.error(f"Error loading history: {str(e)}")


def _render_statistics():
    """Render Top 5 statistics"""
    st.subheader("Statistics")

    try:
        response = requests.get(f"{API_BASE}/top5/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Strategies", data.get('total_strategies', 0))

            with col2:
                st.metric("Total Batches", data.get('total_batches', 0))

            with col3:
                st.metric("Avg Score", f"{data.get('average_score', 0):.4f}")

            # Type distribution
            st.subheader("Type Distribution")
            type_dist = data.get('type_distribution', {})

            if type_dist:
                df = pd.DataFrame([
                    {"Type": k, "Count": v}
                    for k, v in type_dist.items()
                ])
                st.bar_chart(df.set_index('Type'))
            else:
                st.info("No type distribution data")

            # Batch info
            st.write("**Batch Info**")
            st.write(f"- Oldest batch: {data.get('oldest_batch', 'N/A')}")
            st.write(f"- Newest batch: {data.get('newest_batch', 'N/A')}")

    except Exception as e:
        st.error(f"Error loading stats: {str(e)}")


def _load_top5_strategies(batch_id: str = None):
    """Load Top 5 strategies from API"""
    try:
        url = f"{API_BASE}/top5/list"
        if batch_id:
            url += f"?batch_id={batch_id}"

        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            st.session_state.top5_strategies = data.get('strategies', [])
        else:
            st.error(f"Failed to load Top 5: {response.text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")


def _promote_to_validation(strategy_id: str):
    """Promote strategy to validation"""
    try:
        response = requests.post(
            f"{API_BASE}/top5/promote/{strategy_id}",
            timeout=10
        )
        if response.status_code == 200:
            st.success(f"Strategy {strategy_id} ready for 7-Phase validation")
            st.info("Go to Validation tab to run the validation")
        else:
            st.error(f"Failed to promote: {response.text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")


def _validate_all_top5(strategies: List[Dict]):
    """Validate all Top 5 strategies"""
    strategy_ids = [s.get('id') or s.get('strategy_id') for s in strategies]

    try:
        with st.spinner(f"Validating {len(strategy_ids)} strategies..."):
            response = requests.post(
                f"{API_BASE}/validation/batch",
                json={
                    "strategy_ids": strategy_ids,
                    "stop_on_failure": False
                },
                timeout=300
            )

            if response.status_code == 200:
                data = response.json()
                st.success(f"Validation complete: {data['passed']}/{data['validated']} passed")
            else:
                st.error(f"Validation failed: {response.text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")


def _clear_all_top5():
    """Clear all Top 5 strategies"""
    try:
        response = requests.delete(
            f"{API_BASE}/top5/clear?confirm=true",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            st.success(f"Cleared {data['strategies_removed']} strategies")
            st.session_state.top5_strategies = []
            st.rerun()
        else:
            st.error(f"Failed to clear: {response.text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")

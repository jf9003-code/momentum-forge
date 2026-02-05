"""
Generator Institutional Component
UI for strategy generation with batch control
"""
import streamlit as st
import requests
import time
from typing import Dict, List, Optional
import json

# API Base URL
API_BASE = "http://localhost:8000"


def render_generator_panel():
    """Render the strategy generator panel"""
    st.header("Strategy Generator")

    # Initialize session state
    if 'generation_running' not in st.session_state:
        st.session_state.generation_running = False
    if 'generated_strategies' not in st.session_state:
        st.session_state.generated_strategies = []
    if 'selected_for_top5' not in st.session_state:
        st.session_state.selected_for_top5 = []

    # Generator Configuration
    with st.expander("Generator Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            count = st.number_input(
                "Number of Strategies",
                min_value=1,
                max_value=100,
                value=10,
                step=5
            )

            timeframe = st.selectbox(
                "Timeframe",
                options=["", "M5", "M15", "M30", "H1", "H4", "D1"],
                index=0,
                help="Leave empty for all timeframes"
            )

        with col2:
            strategy_types = st.multiselect(
                "Strategy Types",
                options=[
                    "mean_reversion",
                    "momentum",
                    "breakout",
                    "volatility_expansion",
                    "time_based"
                ],
                default=["momentum", "mean_reversion", "breakout"],
                help="Select types to generate"
            )

            market_type = st.selectbox(
                "Market Type",
                options=["", "forex", "indices", "stocks", "crypto", "futures"],
                index=0,
                help="Leave empty for all markets"
            )

        use_existing = st.checkbox(
            "Use existing strategies for clone comparison",
            value=True,
            help="Compare against existing strategies to avoid clones"
        )

    # Generation Controls
    st.subheader("Generation Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Generate (Sync)", type="primary", disabled=st.session_state.generation_running):
            _run_sync_generation(
                count=count,
                strategy_types=strategy_types if strategy_types else None,
                timeframe=timeframe if timeframe else None,
                market_type=market_type if market_type else None,
                use_existing=use_existing
            )

    with col2:
        if st.button("Start (Async)", disabled=st.session_state.generation_running):
            _start_async_generation(
                count=count,
                strategy_types=strategy_types if strategy_types else None,
                timeframe=timeframe if timeframe else None,
                market_type=market_type if market_type else None,
                use_existing=use_existing
            )

    with col3:
        if st.button("Stop", disabled=not st.session_state.generation_running):
            _stop_generation()

    # Progress display
    if st.session_state.generation_running:
        _show_progress()

    # Results display
    if st.session_state.generated_strategies:
        _display_results()


def _run_sync_generation(
    count: int,
    strategy_types: Optional[List[str]],
    timeframe: Optional[str],
    market_type: Optional[str],
    use_existing: bool
):
    """Run synchronous generation"""
    try:
        with st.spinner(f"Generating {count} strategies..."):
            response = requests.post(
                f"{API_BASE}/generator/generate",
                json={
                    "count": count,
                    "strategy_types": strategy_types,
                    "timeframe": timeframe,
                    "market_type": market_type,
                    "use_existing": use_existing
                },
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state.generated_strategies = data.get('strategies', [])
                st.success(
                    f"Generated {data['total_accepted']} strategies "
                    f"({data['total_rejected']} rejected) in {data['duration_ms']:.0f}ms"
                )

                # Show clone filter stats
                stats = data.get('clone_filter_stats', {})
                if stats:
                    st.info(
                        f"Clone Filter: {stats.get('unique', 0)} unique, "
                        f"{stats.get('clones_detected', 0)} clones detected, "
                        f"Avg distance: {stats.get('avg_distance', 0):.4f}"
                    )
            else:
                st.error(f"Generation failed: {response.text}")

    except requests.exceptions.Timeout:
        st.error("Generation timed out. Try reducing the count or use async generation.")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def _start_async_generation(
    count: int,
    strategy_types: Optional[List[str]],
    timeframe: Optional[str],
    market_type: Optional[str],
    use_existing: bool
):
    """Start async generation"""
    try:
        response = requests.post(
            f"{API_BASE}/generator/start",
            json={
                "count": count,
                "strategy_types": strategy_types,
                "timeframe": timeframe,
                "market_type": market_type,
                "use_existing": use_existing
            },
            timeout=10
        )

        if response.status_code == 200:
            st.session_state.generation_running = True
            st.info("Generation started. Monitoring progress...")
            st.rerun()
        else:
            st.error(f"Failed to start: {response.text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")


def _stop_generation():
    """Stop async generation"""
    try:
        response = requests.post(f"{API_BASE}/generator/stop", timeout=5)
        if response.status_code == 200:
            st.session_state.generation_running = False
            st.warning("Generation stopped")
            st.rerun()
    except Exception as e:
        st.error(f"Error stopping: {str(e)}")


def _show_progress():
    """Show generation progress"""
    try:
        response = requests.get(f"{API_BASE}/generator/progress", timeout=5)
        if response.status_code == 200:
            data = response.json()

            if data.get('is_running'):
                progress = data.get('percent', 0) / 100
                st.progress(progress, text=f"Progress: {data.get('current', 0)}/{data.get('total', 0)}")

                # Auto-refresh
                time.sleep(1)
                st.rerun()
            else:
                # Generation finished
                st.session_state.generation_running = False
                _fetch_results()
                st.rerun()

    except Exception as e:
        st.error(f"Error checking progress: {str(e)}")


def _fetch_results():
    """Fetch generation results"""
    try:
        response = requests.get(f"{API_BASE}/generator/result", timeout=10)
        if response.status_code == 200:
            data = response.json()
            st.session_state.generated_strategies = data.get('accepted_strategies', [])
    except Exception:
        pass


def _display_results():
    """Display generated strategies"""
    st.subheader(f"Generated Strategies ({len(st.session_state.generated_strategies)})")

    # Selection for Top 5
    st.write("Select strategies for Top 5:")

    for i, strategy in enumerate(st.session_state.generated_strategies):
        col1, col2, col3, col4 = st.columns([0.5, 2, 2, 1])

        with col1:
            selected = st.checkbox(
                "Select",
                key=f"select_{strategy.get('id', i)}",
                label_visibility="collapsed"
            )
            if selected and strategy['id'] not in st.session_state.selected_for_top5:
                st.session_state.selected_for_top5.append(strategy['id'])
            elif not selected and strategy['id'] in st.session_state.selected_for_top5:
                st.session_state.selected_for_top5.remove(strategy['id'])

        with col2:
            st.write(f"**{strategy.get('id', f'Strategy {i}')}**")
            st.caption(f"Type: {strategy.get('strategy_type', 'N/A')}")

        with col3:
            st.write(f"Template: {strategy.get('template_id', 'N/A')}")
            indicators = strategy.get('indicators', [])
            ind_names = [ind.get('name', '') for ind in indicators if isinstance(ind, dict)]
            st.caption(f"Indicators: {', '.join(ind_names[:3])}")

        with col4:
            distance = strategy.get('_clone_distance', 0)
            st.metric("Distance", f"{distance:.3f}")

        st.divider()

    # Save to Top 5 button
    if st.session_state.selected_for_top5:
        if st.button(
            f"Save {len(st.session_state.selected_for_top5)} to Top 5",
            type="primary"
        ):
            _save_to_top5(st.session_state.selected_for_top5)


def _save_to_top5(strategy_ids: List[str]):
    """Save selected strategies to Top 5"""
    try:
        response = requests.post(
            f"{API_BASE}/top5/save",
            json={"strategy_ids": strategy_ids[:5]},  # Max 5
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            st.success(
                f"Saved {data['strategies_added']} strategies to Top 5 "
                f"(Batch: {data['batch_id']})"
            )
            st.session_state.selected_for_top5 = []
        else:
            st.error(f"Failed to save: {response.text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")


def render_strategy_detail(strategy: Dict):
    """Render detailed view of a strategy"""
    st.subheader(f"Strategy: {strategy.get('id', 'Unknown')}")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Basic Info**")
        st.write(f"- Type: {strategy.get('strategy_type', 'N/A')}")
        st.write(f"- Template: {strategy.get('template_id', 'N/A')}")
        st.write(f"- Batch: {strategy.get('batch_id', 'N/A')}")

        st.write("**Indicators**")
        for ind in strategy.get('indicators', []):
            if isinstance(ind, dict):
                st.write(f"- {ind.get('name', 'Unknown')}: {ind.get('parameters', {})}")

    with col2:
        st.write("**Entry Logic**")
        entry = strategy.get('entry_logic', {})
        st.write(f"Logic: {entry.get('logic_type', 'AND')}")
        for cond in entry.get('conditions', []):
            if isinstance(cond, dict):
                st.write(f"- {cond.get('indicator', '')} {cond.get('operator', '')} {cond.get('compare_to', '')}")

        st.write("**Exit Logic**")
        exit_logic = strategy.get('exit_logic', {})
        st.write(f"Logic: {exit_logic.get('logic_type', 'OR')}")
        for cond in exit_logic.get('conditions', []):
            if isinstance(cond, dict):
                st.write(f"- {cond.get('indicator', '')} {cond.get('operator', '')} {cond.get('compare_to', '')}")

    st.write("**Risk Configuration**")
    risk = strategy.get('risk_config', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("SL (ATR)", f"{risk.get('atr_sl_mult', 'N/A')}")
    with col2:
        st.metric("TP (ATR)", f"{risk.get('atr_tp_mult', 'N/A')}")
    with col3:
        st.metric("Risk/Trade", f"{risk.get('risk_per_trade', 0) * 100:.2f}%")

    # Raw JSON
    with st.expander("Raw JSON"):
        st.json(strategy)

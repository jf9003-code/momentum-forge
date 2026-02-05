"""
D5 Robust Master - Frontend Application
Institutional Trading Strategy Validation Platform

Streamlit-based UI with modular components
"""
import streamlit as st
import requests
from datetime import datetime

# Page config
st.set_page_config(
    page_title="D5 Robust Master",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_BASE = "http://localhost:8000"


def check_backend_connection():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def main():
    """Main application entry point"""
    # Header
    st.title("D5 Robust Master")
    st.caption("Institutional Trading Strategy Validation Platform")

    # Sidebar
    with st.sidebar:
        st.header("Navigation")

        # Backend status
        backend_status = check_backend_connection()
        if backend_status:
            st.success("Backend: Connected")
        else:
            st.error("Backend: Disconnected")
            st.warning("Start the backend with: `python -m backend.app.main`")

        st.divider()

        # Navigation
        page = st.radio(
            "Select Page",
            options=[
                "Generator",
                "Top 5 Strategies",
                "Validation (7-Phase)",
                "Settings"
            ],
            index=0
        )

        st.divider()

        # Quick stats
        if backend_status:
            _render_sidebar_stats()

    # Main content
    if not backend_status:
        st.warning("Backend is not connected. Start the backend server to use the application.")
        st.code("cd /path/to/project && python -m backend.app.main", language="bash")
        return

    # Render selected page
    if page == "Generator":
        _render_generator_page()
    elif page == "Top 5 Strategies":
        _render_top5_page()
    elif page == "Validation (7-Phase)":
        _render_validation_page()
    elif page == "Settings":
        _render_settings_page()


def _render_sidebar_stats():
    """Render quick stats in sidebar"""
    try:
        # Generator stats
        gen_response = requests.get(f"{API_BASE}/generator/stats", timeout=5)
        if gen_response.status_code == 200:
            gen_stats = gen_response.json()
            st.metric("Total Generated", gen_stats.get('total_generated', 0))
            st.metric("Accepted", gen_stats.get('total_accepted', 0))

        # Top 5 stats
        top5_response = requests.get(f"{API_BASE}/top5/stats", timeout=5)
        if top5_response.status_code == 200:
            top5_stats = top5_response.json()
            st.metric("Top 5 Total", top5_stats.get('total_strategies', 0))

    except Exception:
        pass


def _render_generator_page():
    """Render the generator page"""
    from frontend.components.generator_institutional import render_generator_panel
    render_generator_panel()


def _render_top5_page():
    """Render the Top 5 page"""
    from frontend.components.top5_panel import render_top5_panel
    render_top5_panel()


def _render_validation_page():
    """Render the validation page"""
    from frontend.components.validation_panel import render_validation_panel, render_batch_validation

    tab1, tab2 = st.tabs(["Single Strategy", "Batch Validation"])

    with tab1:
        render_validation_panel()

    with tab2:
        render_batch_validation()


def _render_settings_page():
    """Render the settings page"""
    st.header("Settings")

    # API Info
    st.subheader("API Information")
    try:
        response = requests.get(f"{API_BASE}/info", timeout=5)
        if response.status_code == 200:
            info = response.json()

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Application**")
                st.write(f"- Name: {info.get('app_name', 'N/A')}")
                st.write(f"- Version: {info.get('version', 'N/A')}")
                st.write(f"- Debug: {info.get('debug', False)}")

            with col2:
                st.write("**Features**")
                features = info.get('features', {})

                gen_features = features.get('generator', {})
                st.write(f"- Min Structural Distance: {gen_features.get('min_structural_distance', 'N/A')}")
                st.write(f"- Max Batch Size: {gen_features.get('max_batch_size', 'N/A')}")

                val_features = features.get('validation', {})
                st.write(f"- Validation Phases: {val_features.get('phases', 'N/A')}")

    except Exception as e:
        st.error(f"Failed to load API info: {str(e)}")

    # Generator Configuration
    st.subheader("Generator Configuration")

    st.write("**Current Clone Filter Settings:**")
    st.info("""
    - **min_structural_distance**: 0.08 (reduced from 0.15)
    - **parameter_weight**: 2.5 (increased from 2.0)
    - **Same-template boost**: Applied when parameter distance > 0.3

    These settings address the over-rejection issue where same-template
    strategies were being incorrectly marked as clones.
    """)

    # Reset Generator
    st.subheader("Reset Generator")

    col1, col2 = st.columns(2)

    with col1:
        seed = st.number_input("Seed (optional)", min_value=0, value=0)

    with col2:
        if st.button("Reset Generator"):
            try:
                params = {}
                if seed > 0:
                    params['seed'] = seed

                response = requests.post(
                    f"{API_BASE}/generator/reset",
                    params=params,
                    timeout=10
                )
                if response.status_code == 200:
                    st.success("Generator reset successfully")
                else:
                    st.error(f"Reset failed: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Templates
    st.subheader("Available Templates")

    try:
        response = requests.get(f"{API_BASE}/generator/templates", timeout=10)
        if response.status_code == 200:
            data = response.json()
            templates = data.get('templates', [])

            for template in templates:
                with st.expander(f"{template.get('name', 'Unknown')} ({template.get('template_id', 'N/A')})"):
                    st.write(f"**Type:** {template.get('strategy_type', 'N/A')}")
                    st.write(f"**Description:** {template.get('description', 'N/A')}")

                    st.write("**Indicator Slots:**")
                    for slot in template.get('indicator_slots', []):
                        st.write(f"- {slot.get('name', 'N/A')}: {', '.join(slot.get('allowed_indicators', []))}")

                    st.write("**Allowed Timeframes:**")
                    st.write(', '.join(template.get('allowed_timeframes', ['Any'])))

                    st.write("**Allowed Markets:**")
                    st.write(', '.join(template.get('allowed_markets', ['Any'])))

    except Exception as e:
        st.error(f"Failed to load templates: {str(e)}")


# Custom CSS
def apply_custom_css():
    """Apply custom CSS styles"""
    st.markdown("""
    <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }

        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }

        div[data-testid="stExpander"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
        }

        .success-text {
            color: #28a745;
            font-weight: bold;
        }

        .error-text {
            color: #dc3545;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    apply_custom_css()
    main()

"""Interactive Streamlit dashboard for inspecting a trained Double DQN agent.

Run with: streamlit run scripts/demo_app.py
"""
import _bootstrap  # noqa: F401 — adds src/ to sys.path

import sys
from pathlib import Path

# Make the app/ package importable from the repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.helpers import find_checkpoints, load_resources
from app.pages import explorer, race, replay

st.set_page_config(
    page_title="AI Trader — Double DQN",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stMetric { background: #1a1f2b; padding: 12px; border-radius: 8px; border: 1px solid #3a3f4b; }
    .stMetric label { color: #9da5b4 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e6eaf0 !important; }
    div[data-testid="stSidebar"] { background: #0e1117; }
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.title("📈 AI Trader")
st.sidebar.markdown("**Double DQN Trading Agent**")

page = st.sidebar.radio(
    "Navigate",
    [
        "🎬 Live Agent Replay",
        "🏁 Agent vs Buy & Hold",
        "🎛️ Hyperparameter Explorer",
    ],
)

ckpt_map = find_checkpoints()
cfg, bundle = load_resources()

if page == "🎬 Live Agent Replay":
    replay.render(cfg, bundle, ckpt_map)
elif page == "🏁 Agent vs Buy & Hold":
    race.render(cfg, bundle, ckpt_map)
elif page == "🎛️ Hyperparameter Explorer":
    explorer.render()

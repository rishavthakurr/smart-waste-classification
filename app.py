"""Smart Waste Classification - Streamlit entry point.  Run:  streamlit run app.py"""
import streamlit as st

from src.database import init_db
from src.ui import apply_theme

st.set_page_config(page_title="Smart Waste Classifier", page_icon=":material/recycling:", layout="wide")
init_db()
apply_theme()

pages = [
    st.Page("pages/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page("pages/classify_waste.py", title="Classify Waste", icon=":material/photo_camera:"),
    st.Page("pages/live_detection.py", title="Live Detection", icon=":material/videocam:"),
    st.Page("pages/model_performance.py", title="Model Performance", icon=":material/monitoring:"),
    st.Page("pages/history.py", title="Classification History", icon=":material/history:"),
    st.Page("pages/about.py", title="About the Project", icon=":material/info:"),
]
st.navigation(pages).run()

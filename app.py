"""Streamlit front end. The Groq API key is read only from Streamlit secrets."""
import os

# Turn off CrewAI telemetry (optional). Must be set before importing crewai.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import streamlit as st

from research_agent import run_research

st.set_page_config(page_title="AI Research Agent", page_icon="🔎")
st.title("🔎 AI Research Agent")
st.caption("CrewAI + Groq (gpt-oss-120b) + DuckDuckGo search")

# Read the key from Streamlit secrets (App settings > Secrets on Streamlit Cloud)
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error(
        "GROQ_API_KEY is missing. In Streamlit Cloud open your app's "
        "Settings > Secrets and add:  GROQ_API_KEY = \"your-key\""
    )
    st.stop()

topic = st.text_input("Research topic", placeholder="e.g. Solid-state batteries in 2026")

if st.button("Write report", type="primary"):
    if not topic.strip():
        st.error("Please enter a topic.")
    else:
        with st.spinner("Researching and writing... this can take 1-2 minutes."):
            try:
                st.session_state["report"] = run_research(topic.strip(), api_key)
            except Exception as e:
                st.error(f"Something went wrong: {e}")

if "report" in st.session_state:
    st.markdown(st.session_state["report"])
    st.download_button(
        "Download report (.md)",
        data=st.session_state["report"],
        file_name="research_report.md",
        mime="text/markdown",
    )

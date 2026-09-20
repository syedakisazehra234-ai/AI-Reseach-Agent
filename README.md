# AI Research Agent

Single-agent research app: enter a topic, get a sourced Markdown report.

Stack: CrewAI, Groq (`openai/gpt-oss-120b`), DuckDuckGo search (`ddgs`), Streamlit.

## Deploy on Streamlit Cloud
1. Push this repo to GitHub.
2. On share.streamlit.io choose **Create app**, select the repo, main file `app.py`.
3. In **Advanced settings** choose **Python 3.12** and add this under **Secrets**:
   ```toml
   GROQ_API_KEY = "your-groq-key"
   ```
4. Deploy.

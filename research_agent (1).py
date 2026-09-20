"""The CrewAI part of the app: one agent, one task, one crew."""
import os
import sys
import time

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS

MODEL = "groq/openai/gpt-oss-120b"  # "groq/" tells CrewAI which provider to use


def _disable_cache_breakpoint():
    """Workaround for a known CrewAI bug: some versions add a 'cache_breakpoint'
    field to messages (meant for Anthropic), and Groq rejects it with a 400 error.
    This removes that field. It is harmless if your CrewAI version is already fixed."""
    # 1) Make the marking function do nothing everywhere it was imported
    for name, mod in list(sys.modules.items()):
        if name.startswith("crewai") and mod is not None:
            try:
                if hasattr(mod, "mark_cache_breakpoint"):
                    setattr(mod, "mark_cache_breakpoint", lambda msg, *a, **k: msg)
            except Exception:
                pass

    # 2) Also strip the field from every message just before it is sent
    original = getattr(LLM, "_format_messages_for_provider", None)
    if original is None or getattr(original, "_patched", False):
        return

    def patched(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if isinstance(result, list):
            return [
                {k: v for k, v in m.items() if k != "cache_breakpoint"}
                if isinstance(m, dict) else m
                for m in result
            ]
        return result

    patched._patched = True
    LLM._format_messages_for_provider = patched


_disable_cache_breakpoint()


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo. Returns titles, links and short summaries."""
    try:
        results = DDGS().text(query, max_results=4)
    except Exception as e:  # DuckDuckGo can rate-limit; tell the agent instead of crashing
        return f"Search failed: {e}. Try a different or shorter query."
    if not results:
        return "No results found."
    return "\n\n".join(
        f"Title: {r.get('title')}\nURL: {r.get('href')}\nSummary: {(r.get('body') or '')[:300]}"
        for r in results
    )


def run_research(topic: str, groq_api_key: str) -> str:
    """Runs the research agent on a topic and returns the report as Markdown."""
    os.environ["GROQ_API_KEY"] = groq_api_key

    llm = LLM(model=MODEL, temperature=0.3, max_tokens=3000, reasoning_effort="low")

    researcher = Agent(
        role="Senior Research Analyst",
        goal="Research a topic on the web and write an accurate, well-organized report.",
        backstory=(
            "You are a careful analyst. You search the web, compare several sources, "
            "and never invent facts or links. If sources disagree or information is "
            "missing, you say so."
        ),
        tools=[web_search],
        llm=llm,
        allow_delegation=False,
        max_iter=6,  # limits how many search/think loops the agent may do
        verbose=True,
    )

    task = Task(
        description=(
            "Research the topic: {topic}\n"
            "Use the web_search tool 2 or 3 times with different queries (no more than 3). "
            "Then write a report based only on what you found."
        ),
        expected_output=(
            "A Markdown report with: a title, a short summary, 3-5 sections with key "
            "findings, a conclusion, and a 'Sources' list with URLs you actually used."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    # Groq's free tier allows only ~8000 tokens per minute. If we hit that limit,
    # wait for the counter to reset and try again instead of failing.
    attempts = 4
    for attempt in range(attempts):
        try:
            return crew.kickoff(inputs={"topic": topic}).raw
        except Exception as e:
            text = str(e).lower()
            is_rate_limit = "rate limit" in text or "ratelimit" in text
            if is_rate_limit and attempt < attempts - 1:
                time.sleep(20 * (attempt + 1))  # waits 20s, 40s, 60s
                continue
            raise

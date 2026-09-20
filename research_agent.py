"""The CrewAI part of the app: one agent, one task, one crew."""
import os

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS

MODEL = "groq/openai/gpt-oss-120b"  # "groq/" tells CrewAI which provider to use


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo. Returns titles, links and short summaries."""
    try:
        results = DDGS().text(query, max_results=6)
    except Exception as e:  # DuckDuckGo can rate-limit; tell the agent instead of crashing
        return f"Search failed: {e}. Try a different or shorter query."
    if not results:
        return "No results found."
    return "\n\n".join(
        f"Title: {r.get('title')}\nURL: {r.get('href')}\nSummary: {r.get('body')}"
        for r in results
    )


def run_research(topic: str, groq_api_key: str) -> str:
    """Runs the research agent on a topic and returns the report as Markdown."""
    os.environ["GROQ_API_KEY"] = groq_api_key

    llm = LLM(model=MODEL, temperature=0.3, max_tokens=4000)

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
        max_iter=8,  # limits how many search/think loops the agent may do
        verbose=True,
    )

    task = Task(
        description=(
            "Research the topic: {topic}\n"
            "Use the web_search tool several times with different queries. "
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

    result = crew.kickoff(inputs={"topic": topic})
    return result.raw

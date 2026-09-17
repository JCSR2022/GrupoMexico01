"""Optional AI-agent integration for the production ETL workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_groq import ChatGroq


SYSTEM_PROMPT = """\
You are an AI agent specialized in ETL, data quality assessment, cleansing,
and preparation. Always inspect the dataset before proposing transformations.
Never assume column names, types, formats, or business rules. Use only the
available controlled tools. Distinguish issues identified, transformations
proposed, transformations executed, and transformations validated. Every
data-quality issue must be recorded in the data-quality log.
"""


def _load_environment_file(path: Path = Path(".env")) -> None:
    """Load only valid KEY=VALUE entries without dotenv parser warnings."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("\"'")


def create_etl_agent(tools: list[Any], model: str | None = None) -> Any:
    """Create a Groq-backed agent using credentials from the environment."""
    _load_environment_file()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Set it in the environment or create "
            "a .env file from .env.example before starting the agent."
        )
    llm = ChatGroq(
        model=model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        max_tokens=2048,
        api_key=api_key,
    )
    try:
        from langchain.agents import create_agent
    except ImportError:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        tool_calling_agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=tool_calling_agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
        )
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


__all__ = ["SYSTEM_PROMPT", "create_etl_agent"]

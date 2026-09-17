"""Interactive entry point for the AI-assisted ETL workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent import create_etl_agent
from tools import (
    ETLContext,
    create_tools,
    detect_missing_values,
    detect_outliers,
    inspect_dataset,
    save_df,
    save_etl_context_log,
)


def run_agent(
    user_input: str,
    history: list[BaseMessage],
    agent: Any,
) -> dict[str, Any]:
    """Send one user message to the agent and update conversation history."""
    messages = history + [HumanMessage(content=user_input)]
    try:
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50},
        )
    except (KeyError, TypeError):
        # Compatibility with the older AgentExecutor returned by LangChain 0.3.
        result = agent.invoke({"input": user_input, "chat_history": history})

    result_messages = result.get("messages")
    if result_messages:
        history.clear()
        history.extend(result_messages)
        response = result_messages[-1]
        content = getattr(response, "content", str(response))
    else:
        content = str(result.get("output", result))
        history.extend([HumanMessage(content=user_input), AIMessage(content=content)])

    return {"messages": history, "content": content}


def run_deterministic_request(user_input: str, context: ETLContext) -> str | None:
    """Handle common read-only requests without relying on model output parsing."""
    normalized = re.sub(r"\s+", " ", user_input.casefold()).strip()

    if (
        "inspect" in normalized
        or "list columns" in normalized
        or "column names" in normalized
        or normalized in {"columns", "show columns"}
    ):
        profile = inspect_dataset(context)
        columns = [column["name"] for column in profile["columns"]]
        return (
            f"Dataset: {profile['shape']['rows']} rows and "
            f"{profile['shape']['columns']} columns.\n"
            "Columns:\n"
            + "\n".join(f"{index}. {column}" for index, column in enumerate(columns, 1))
        )

    if "outlier" in normalized:
        return json.dumps(detect_outliers(context), ensure_ascii=False, indent=2)

    if "missing" in normalized:
        return json.dumps(detect_missing_values(context), ensure_ascii=False, indent=2)

    return None


def run_interactive_etl(input_path: Path, output_dir: Path) -> ETLContext:
    """Load a dataset and keep an agent-driven ETL session open until exit."""
    context = ETLContext(pd.read_excel(input_path), output_dir=output_dir)
    tools = create_tools(context)
    agent = create_etl_agent(tools)
    history: list[BaseMessage] = []

    print("ETL agent ready. Type 'quit', 'exit', or 'q' to finish.")
    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in {"quit", "exit", "q", ""}:
                break

            print("Agent: ", end="", flush=True)
            try:
                deterministic_response = run_deterministic_request(user_input, context)
                if deterministic_response is not None:
                    print(deterministic_response)
                else:
                    response = run_agent(user_input, history, agent)
                    content = response["content"]
                    if (
                        "parsing failed" in content.casefold()
                        or "could not be parsed" in content.casefold()
                    ):
                        fallback_response = run_deterministic_request(user_input, context)
                        print(fallback_response or content)
                    else:
                        print(content)
            except Exception as exc:
                fallback_response = run_deterministic_request(user_input, context)
                if fallback_response is not None:
                    print(fallback_response)
                else:
                    print(f"Error: Agent request failed: {exc}")
            print()
    finally:
        save_df(context)
        save_etl_context_log(context)
        print("Goodbye!")

    return context


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an interactive ETL agent.")
    parser.add_argument("--input", type=Path, required=True, help="Input Excel file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for generated datasets and logs.",
    )
    args = parser.parse_args()
    run_interactive_etl(args.input, args.output_dir)


if __name__ == "__main__":
    main()

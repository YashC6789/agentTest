# src/agent/main_agent.py
from __future__ import annotations

from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from src.agent.setup import build_tool_agent
from src.config.prompts import SYSTEM_PROMPT_TEXT


def build_main_agent(tools: List[Dict[str, Any]]):
    """Build the main LangGraph tool-calling agent for evaluation or deployment."""
    return build_tool_agent(tools)


def run_agent_once(agent_executor, user_prompt: str, extra_system_text: str = ""):
    system_content = SYSTEM_PROMPT_TEXT
    if extra_system_text:
        system_content = SYSTEM_PROMPT_TEXT + "\n\n" + extra_system_text

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_prompt),
    ]
    state = agent_executor.invoke({"messages": messages})
    return state["messages"]
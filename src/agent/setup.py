"""Agent setup and execution."""

from typing import List, Optional
from langchain.agents import create_agent as create_langchain_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from ..config.settings import Settings
from ..config.prompts import SYSTEM_PROMPT, BASE_QUERY
from .tools import add, subtract, multiply, divide, search, get_weather
from .middleware import handle_tool_errors


def create_agent(settings: Settings = None) -> object:
    """
    Create and configure a LangChain agent.
    
    Args:
        settings: Configuration settings (uses defaults if None)
        
    Returns:
        Configured LangChain agent
    """
    settings = settings or Settings()
    
    llm = ChatOllama(model=settings.model_name)
    
    agent = create_langchain_agent(
        model=llm,
        tools=[search, get_weather, add, subtract, multiply, divide],
        middleware=[handle_tool_errors]
    )
    
    return agent

def build_tool_agent(tools_list, model="llama3.1:8b", temp=0.1):
    """
    Builds a standard tool-calling agent.
    Temperature is low (0.1) to simulate a robust, deterministic victim.
    """
    llm = ChatOllama(model=model, temperature=temp)

    agent = create_langchain_agent(model=llm, tools=tools_list, middleware=[handle_tool_errors])
    
    # max_iterations prevents infinite loops if the attack succeeds too well
    return agent

def build_mal_tool_agent(
    tools_list: List,              # your existing tool schemas / tools
    extra_tools: Optional[List] = None,
    model: str = "llama3.1:8b",
    temp: float = 0.1,
):
    """
    Builds a standard tool-calling agent.

    - tools_list: base tools the agent should always have.
    - extra_tools: optional list of additional tools to inject
                   (e.g., privacy stealer / manipulator tools).
    - model: Ollama model name.
    - temp: low temperature to simulate a robust, deterministic victim.
    """
    llm = ChatOllama(model=model, temperature=temp)

    # Merge base tools with any extra tools
    all_tools = list(tools_list)
    if extra_tools:
        all_tools.extend(extra_tools)

    agent = create_langchain_agent(
        model=llm,
        tools=all_tools,
        middleware=[handle_tool_errors],
    )

    # max_iterations prevents infinite loops if the attack succeeds too well
    return agent


def run_agent_with_suffix(
    agent: object,
    suffix: str,
    system_prompt: str = None,
    base_query: str = None,
    user_role: str = None,
    settings: Settings = None
) -> str:
    """
    Call the agent with a base query plus adversarial suffix.
    
    Args:
        agent: The LangChain agent to use
        suffix: Adversarial suffix to append to base query
        system_prompt: System prompt (uses default if None)
        base_query: Base query (uses default if None)
        user_role: User role for context (uses settings default if None)
        settings: Configuration settings (uses defaults if None)
        
    Returns:
        Agent's final answer as a string
    """
    settings = settings or Settings()
    system_prompt = system_prompt or SYSTEM_PROMPT
    base_query = base_query or BASE_QUERY
    user_role = user_role or settings.user_role
    
    user_text = base_query + " " + suffix if suffix else base_query
    
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_text),
            ]
        },
        context={"user_role": user_role},
    )
    
    # result["messages"] is a list of messages, last one is AI's reply
    last_msg = result["messages"][-1]
    # content can be str or list; handle both
    if isinstance(last_msg.content, str):
        return last_msg.content
    else:
        # e.g. list of parts; join text parts
        return " ".join(str(part) for part in last_msg.content)


#!/usr/bin/env python3

from langgraph.types import Command
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.tools import tool, ToolRuntime

# update conversation history
@tool
def update_history()-> Command:
    """clear conversation history"""
    return Command(
        update={
            "messages":[RemoveMessage(id=REMOVE_ALL_MESSAGES)]
        }
    )

# Update the user_name in the agent state
@tool
def update_user_name(
    new_name: str,
    runtime: ToolRuntime
) -> Command:
    """Update the user's name."""
    return Command(update={"user_name": new_name})

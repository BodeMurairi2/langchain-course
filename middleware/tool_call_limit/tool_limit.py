#!/usr/bin/env python3

import os
import random
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents.middleware import ToolCallLimitMiddleware

from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

gemini_config = {
    "gemini_model":os.getenv("GEMINI_AI_MODEL"),
    "gemini_api_key":os.getenv("GEMINI_API_KEY_V")
}

if not all([gemini_config["gemini_model"],gemini_config["gemini_api_key"]]):
    raise ValueError("No APIs configured")

llm = ChatGoogleGenerativeAI(
    model=gemini_config["gemini_model"],
    api_key=gemini_config["gemini_api_key"],
    max_tokens=1500,
    temperature=0.5
    )

memory_config = {
    "memory":InMemorySaver(),
    "thread_id":random.randrange(1,999999)
}

agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=InMemorySaver(),
    middleware=[
        # global limit
        ToolCallLimitMiddleware(run_limit=5, thread_limit=10),
        # specific
        ToolCallLimitMiddleware(
            tool_name="my_tool",
            thread_limit=5,
            run_limit=2,
            exit_behavior="end"
            )
    ]
    )

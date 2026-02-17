#!/usr/bin/env python3

import os
import re
from dotenv import load_dotenv
from uuid import uuid5

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents.middleware import PIIMiddleware

from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_AI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
    max_tokens=1500
    )

memory_config = {
    "memory":InMemorySaver(),
    "thread_id":uuid5()
}

agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=InMemorySaver(),
    middleware=[
        PIIMiddleware("email", strategy="hash"),
        PIIMiddleware("credit_card", strategy="mask"),
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block")
    ]
    )

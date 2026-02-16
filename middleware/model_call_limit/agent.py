#!/usr/bin/env python3

import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from langchain.agents.middleware import ModelCallLimitMiddleware

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_AI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY_V"),
    max_tokens=1500,
    temperature=0.5
    )

agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=5,
            thread_limit=15,
            exit_behavior="error",
            error_message="Number of requests exceeded. Try later"
        )
    ],
    system_prompt="You are a helpful agent who guides customers"
)

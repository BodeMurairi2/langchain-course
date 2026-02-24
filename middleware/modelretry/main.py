#!/usr/bin/env python3

import os
import uuid
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

from langchain_google_genai import GoogleGenerativeAI

from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

gemini_config = {
    "gemini_model":os.getenv("GEMINI_AI_MODEL"),
    "gemini_api_key":os.getenv("GEMINI_API_KEY")
    }

if not all([gemini_config["gemini_model"],gemini_config["gemini_api_key"]]):
    raise ValueError("Gemini API key configuration missing")

llm = GoogleGenerativeAI(
    model=gemini_config["gemini_model"],
    api_key=gemini_config["gemini_api_key"],
    temperature=0.5,
    max_tokens=1000
    )

memory_config = {
    "memory":InMemorySaver(),
    "thread_id":uuid.uuid4()
}

agent = create_agent(
    model=llm,
    tools=["search","find","develop"],
    middleware=[
        ModelRetryMiddleware(
            max_retries=5,
            tools=["search","find"],
            retry_on=("ToolError","ToolException","ToolTimeout"),
            on_failure="continue",
            backoff_factor=2.0,
            initial_delay=2,
            max_delay=80,
            jitter=True
        )
    ]
    )

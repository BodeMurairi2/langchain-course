#!/usr/bin/env python3

import os
import uuid
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage, AIMessage

from langchain.agents.middleware import ModelFallbackMiddleware
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

gemini_config = {
    "gemini_model":os.getenv("GEMINI_AI_MODEL"),
    "gemini_api_key":os.getenv("GEMINI_API_KEY")
    }

if not all([gemini_config["gemini_model"],gemini_config["gemini_api_key"]]):
    raise ValueError("Gemini API key configuration missing")

llm = ChatGoogleGenerativeAI(
    model=gemini_config["gemini_model"],
    api_key=gemini_config["gemini_api_key"],
    temperature=0.5,
    max_tokens=1000
    )

llm_2 = ChatGoogleGenerativeAI(
    model=gemini_config["gemini_model"],
    api_key=gemini_config["gemini_api_key_V"],
    temperature=0.7,
    max_tokens=2500
    )
# memory config
memory_config = {
    "memory":InMemorySaver(),
    "thread_id":uuid.uuid5()
}

agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=memory_config["memory"],
    middleware=[
        ModelFallbackMiddleware([llm_2])
    ],
    system_prompt="You are an helpful customer agent assistant"
)

response = agent.invoke(
    {
        "messages":[HumanMessage(content="How to get free credits to your plateform?")]
    },
    config={
        "configurable":{
            "thread_id":memory_config["thread_id"]
        }
        }
    )

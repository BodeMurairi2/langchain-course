#!/usr/bin/env python3

import os
import uuid
from dotenv import load_dotenv
from pathlib import Path
from pprint import pprint

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from langchain.agents.middleware import SummarizationMiddleware

from langgraph.checkpoint.memory import InMemorySaver
#from langgraph.config import get_stream_writer

from tool.call_tool import get_instant_weather


# Load environment variables
load_dotenv()

# Memory config
memory_config = {
    "MEMORY": InMemorySaver(),
    "THREAD_ID": str(uuid.uuid4())
}

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_AI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
    max_tokens=4000
)

# Tools
tools = [get_instant_weather]

# System prompt
SYS_PROMPT = SystemMessage(
    content=(
        "You are an expert meteorologist. "
        "Only call the weather tool to get real-time data when the user explicitly asks "
        "about weather conditions, temperature, rain, or forecasts. "
        "Do not guess weather conditions. "
        "When advising foreigners, consider local lifestyle, "
        "transport, rain patterns, and health comfort. "
        "Keep advice practical and culturally appropriate. "
        "If the weather tool returns an error, politely explain that "
        "real-time data is unavailable and provide general seasonal advice instead. "
        "Do not use asterisks or ** in your response."
    )
)

# Create agent
weather_agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYS_PROMPT,
    checkpointer=memory_config["MEMORY"],
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens",3000),
            keep=("messages",10)
        )
    ]
)

'''
stream agent progress
Use stream_mode = "updates"

for chunk in weather_agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in Kigali"}]},
    config={"configurable": {"thread_id": memory_config["THREAD_ID"]}},
    stream_mode="updates"
):
    model_output = chunk.get("model")
    if not model_output:
        continue

    for messages in model_output.values():
        for msg in messages:
            if hasattr(msg, "content"):
                print(msg.content)
'''

'''
stream token inside LLM as they are generated
use stream_mode = "messages"
for token, metadata in weather_agent.stream(
    {"messages":[{"role":"user","content":"What is the best cloths to wear with Kigali's current weather conditions"}]},
    config={"configurable":{"thread_id":memory_config["THREAD_ID"]}},
    stream_mode="messages"
    ):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token.content_blocks}")
    print("\n")
'''

'''
Stream updates from tools as they are being executed
use stream_modes = "custom"
import:
from langgraph.config import get_stream_writer
use the get_stream_writer() to write each execution step

for chunk in weather_agent.stream(
    {"messages":[{"role":"user","content":"What is the best cloths to wear with Kigali's current weather conditions"}]},
    config={"configurable":{"thread_id":memory_config["THREAD_ID"]}},
    stream_mode="custom"
    ):
    print(chunk)
'''

'''
streaming multiple nodes: Agent progress, llm token generation, tool calls executation
stream_mode = []
for stream_mode, chunk in weather_agent.stream(
    {"messages":[{"role":"user","content":"What is the best cloths to wear with Kigali's current weather conditions"}]},
    config={"configurable":{"thread_id":memory_config["THREAD_ID"]}},
    stream_mode=["updates","custom"]
    ):
    print(f"Stream mode: {stream_mode}")
    print("________________________")
    print(f"Chunk: {chunk}")
'''

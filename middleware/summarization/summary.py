#!/usr/bin/env python3

import os
import random
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents.middleware import SummarizationMiddleware

from langgraph.checkpoint.memory import InMemorySaver

# env variables
load_dotenv()

gemini_config = {
    "gemini_ai_model":os.getenv("GEMINI_AI_MODEL"),
    "gemini_ai_key":os.getenv("GEMINI_API_KEY_V")
}

if not all([gemini_config["gemini_ai_model"],gemini_config["gemini_ai_key"]]):
    raise KeyError("No API Key configuration found")

llm = ChatGoogleGenerativeAI(
    model=gemini_config["gemini_ai_model"],
    api_key=gemini_config["gemini_ai_key"],
    max_tokens=1500,
    temperature=0.5
    )

# memory
memory_config = {
    "memory": InMemorySaver(),
    "thread_id": str(random.randrange(1, 9999999))
    }

agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=memory_config["memory"],
    middleware=[
        SummarizationMiddleware(
            model=llm,
            trigger=[("messages", 2)],
            keep=("messages",5)
        )
    ],
    system_prompt="You are helpful customers care assistant agent"
    )

for conv in range(7):
    agent.invoke(
        {
            "messages":[
                HumanMessage(
                    content=f"How to get free discount in your compagny?\nUser_{conv}"
                    )
                    ]
        },
        config={
            "configurable":{
                "thread_id":memory_config["thread_id"]
                }
                }
        )

state = memory_config["memory"].get({
    "configurable": {
        "memory": memory_config["memory"],
        "thread_id": memory_config["thread_id"]
    }
})

print("\n--- RAW MEMORY STATE ---\n")
print(f"{state}\n")

messages = state["channel_values"]["messages"]
for msg in messages:
    print("\n")
    print(type(msg), msg.content)
    print("\n")

for msg in messages:
    print(type(msg), msg.content)

#!/usr/bin/env python3

"""
This script implements an multimodal agent with content block
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.messages import HumanMessage

env_path = Path(__file__).parent.parent / "model" / "auth.env"
load_dotenv(env_path)

model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_AI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.5,
    max_tokens=4000
)

tools = []

# Create the agent
agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

human_message = HumanMessage(content=[
    {"type": "text", "text": "Can you tell me more about this car?"},
    {
        "type": "image",
        "url": "https://www.bentleymotors.com/content/dam/bm/websites/bmcom/bentleymotors-com/homepage/26my-azure/GT%20Azure%20Dynamic%20Desktop.jpg/_jcr_content/renditions/original.image_file.1286.643.file/GT%20Azure%20Dynamic%20Desktop.jpg"
    }
])

response = agent.invoke({"input": human_message})

print(response["output"])

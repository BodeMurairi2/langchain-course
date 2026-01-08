#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from pathlib import Path
from langchain.agents import create_agent
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from start_tool import search_database

env_path = Path(__file__).parent.parent.parent / "model" / "auth.env"
load_dotenv(env_path)

model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_AI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
    max_tokens=4000
)

tools = [search_database]

agent = create_agent(model=model, tools=tools)

prompt = {"messages": [("user", "Search the entire database for the word 'AI' or 'artificial intelligence' . If you find, explain what the world is and its purpose")]}

print(agent.invoke(prompt)["messages"][-1].content)

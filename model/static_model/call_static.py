#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from tool.tool_begin.tool_calling import search, get_weather

load_dotenv()

API_KEY:str = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise FileNotFoundError("No API KEY")

os.environ["GOOGLE_API_KEY"] = API_KEY

def chat_function(message:str)->str:
    model = init_chat_model(f"google_genai:{os.environ.get('GEMINI_AI_MODEL')}")
    response = model.invoke(message)
    return response.text

def init_weather_agent():
    """Initializes a chat agent with weather and search tools."""
    model_name = os.environ.get("GEMINI_AI_MODEL")
    agent = init_chat_model(f"google_genai:{model_name}",
                            tools=[search, get_weather])
    return agent

if __name__ == "__main__":
    message = input("Enter your question")
    print(chat_function(message=message))

    agent = init_weather_agent()
    response = agent.invoke("What's the weather in Kigali?")
    print(response.text)

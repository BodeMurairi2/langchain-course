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

from tool.call_tool import get_instant_weather
from structured_output.output import Response_output, structured_output

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

# Create agent
weather_agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=memory_config["MEMORY"],
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens",3000),
            keep=("messages",10)
        )
    ],
    response_format=Response_output
)

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

# Function to display weather neatly
def display_weather(result: dict):
    """
    Nicely formats the weather API response
    """
    if not result or "error" in result:
        print("Weather data unavailable. Showing general seasonal advice instead.")
        return
    
    print("\n🌤 Weather Report 🌤")
    print("------------------------")
    print(f"Location: {result.get('location', 'Unknown')}")
    print(f"Local Time: {result.get('localtime', 'Unknown')}")
    print(f"Temperature: {result.get('temperature_c', 'Unknown')} °C")
    
    condition = result.get("condition", {}).get("text") if result.get("condition") else "Unknown"
    print(f"Condition: {condition}")
    
    print(f"Humidity: {result.get('humidity', 'Unknown')}%")
    print(f"Wind: {result.get('wind_kph', 'Unknown')} kph, Direction: {result.get('wind_dir', 'Unknown')}")
    
    print(f"Feels Like: {result.get('feelslike_c', 'Unknown')} °C")
    print("------------------------\n")


# Main agent loop
def agent(agent):
    print("Hello! I am your Meteorologist assistant. What can I do for you today?")
    while True:
        user_question = input("Type your question here...\n")
        
        if user_question.lower() in ["exit", "quit"]:
            print("Goodbye! Stay safe and enjoy the weather.")
            break

        if not user_question.strip():
            print("No question asked! Please enter your question...")
            continue
        custom_profile = {
            "structured_ouput":True
        }
        agent_response = weather_agent.invoke(
            {"messages": [SYS_PROMPT, HumanMessage(content=user_question)]},
            config={"configurable": {"thread_id": memory_config["THREAD_ID"], **custom_profile}}
        )

        ai_response = agent_response["messages"][-1].content

        if isinstance(ai_response, list):
            for message_dict in ai_response:
                content = message_dict.get("text") or message_dict.get("content")
                if isinstance(content, dict) and "temperature_c" in content:
                    display_weather(content)
                else:
                    print("\nYour Assistant:\n------------------------")
                    pprint(content)
                    print("------------------------\n")
        else:
            if isinstance(ai_response, dict) and "temperature_c" in ai_response:
                display_weather(ai_response)
            else:
                print("\nYour Assistant:\n------------------------")
                pprint(ai_response)
                print("------------------------\n")


if __name__ == "__main__":
    agent(agent=weather_agent)

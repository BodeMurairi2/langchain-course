#!/usr/bin/env python3

import os
import uuid
from dotenv import load_dotenv
from pprint import pprint

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from tool.call_tool import get_instant_weather

load_dotenv()

structured_output_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "location_time": {"type": "string"},
        "temperature_celsius": {"type": "number"},
        "pressure_mb": {"type": "number"},
        "precip_mm": {"type": "number"},
        "humidity": {"type": "number"},
        "feellike_c": {"type": "number"},
        "last_City": {"type": "string"},
        "ai_response": {"type": "string"}
    },
    "required": [
        "location",
        "location_time",
        "temperature_celsius",
        "pressure_mb",
        "precip_mm",
        "humidity",
        "feellike_c",
        "last_City",
        "ai_response"
    ]
}

memory = InMemorySaver()
thread_id = str(uuid.uuid4())

model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_AI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
    max_tokens=4000
)

tools = [get_instant_weather]

weather_agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=memory,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 3000),
            keep=("messages", 10)
        )
    ],
    response_format={
        "type": "json_schema",
        "json_schema": structured_output_schema
    }
)

SYS_PROMPT = SystemMessage(
    content=(
        "You are an expert meteorologist. "
        "Only call the weather tool when the user explicitly asks about weather conditions, "
        "temperature, rain, humidity, pressure, or forecasts. "
        "Never guess weather conditions. "
        "Always return your response strictly following the provided JSON schema. "
        "If real-time data is unavailable, explain politely in ai_response and provide seasonal advice. "
        "Do not use markdown formatting symbols."
    )
)

def display_weather(result: dict):
    print("\n🌤 Weather Report 🌤")
    print("-----------------------------")
    print(f"Location: {result.get('location')}")
    print(f"Local Time: {result.get('location_time')}")
    print(f"Temperature: {result.get('temperature_celsius')} °C")
    print(f"Pressure: {result.get('pressure_mb')} mb")
    print(f"Precipitation: {result.get('precip_mm')} mm")
    print(f"Humidity: {result.get('humidity')} %")
    print(f"Feels Like: {result.get('feellike_c')} °C")
    print("-----------------------------")
    print("Advice:")
    print(result.get("ai_response"))
    print("-----------------------------\n")

def run_agent():
    print("Hello! I am your Meteorologist assistant.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_question = input("Ask your question:\n> ")

        if user_question.lower() in ["exit", "quit"]:
            print("Goodbye! Stay safe and enjoy the weather.")
            break

        if not user_question.strip():
            print("Please enter a valid question.\n")
            continue

        try:
            response = weather_agent.invoke(
                {"messages": [SYS_PROMPT, HumanMessage(content=user_question)]},
                config={"configurable": {"thread_id": thread_id}}
            )

            structured_result = response.get("output")

            if isinstance(structured_result, dict):
                display_weather(structured_result)
            else:
                print("Unexpected response format:")
                pprint(structured_result)

        except Exception as e:
            print(f"An error occurred: {e}\n")

# Run Application
if __name__ == "__main__":
    run_agent()

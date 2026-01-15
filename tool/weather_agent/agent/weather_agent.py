#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from pathlib import Path

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from tool.call_tool import get_instant_weather

env_path = Path(__file__).parent.parent/"weather.env"
load_dotenv(env_path)

memory_config = {
    "MEMORY":InMemorySaver(),
    "THREAD_ID":"weather-session-1"
}

model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_AI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
    max_tokens=2000
)

tools = [get_instant_weather]

weather_agent = create_agent(model=model,
                     tools=tools,
                     checkpointer=memory_config["MEMORY"]
                     )

SYS_PROMPT = SystemMessage(
    content=(
        "You are an expert meteorologist"
        "You MUST call the weather tool to get real-time data "
        "before answering any weather-related question. "
        "Do not guess weather conditions. "
        "When advising foreigners, consider local lifestyle, "
        "transport, rain patterns, and health comfort. "
        "Keep advice practical and culturally appropriate. "
        "If the weather tool returns an error, politely explain that "
        "real-time data is unavailable and provide general seasonal advice instead. "
        "Do not use asterisks in your response."
        )
        )

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

        agent_response = weather_agent.invoke(
            {"messages": [SYS_PROMPT, HumanMessage(content=user_question)]},
            config={"configurable":{"thread_id":memory_config["THREAD_ID"]}}
            )

        ai_response = agent_response["messages"][-1].content
        
        if isinstance(ai_response, list):
            for message_dict in ai_response:
                print("Your Assistant:", message_dict["text"])
        
        else:
            print("Your Assistant:", ai_response)

if __name__ == "__main__":
    agent(agent=weather_agent)

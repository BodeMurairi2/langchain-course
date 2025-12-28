#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# create an agent
agent = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_AI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY")
)

# messsages
system_messages = SystemMessage(content="Answer clearly without any ambuguity")

user_input = input("Enter your message\n")
messages = [
    system_messages, HumanMessage(content=user_input)
]

print(agent.invoke(user_input).text)

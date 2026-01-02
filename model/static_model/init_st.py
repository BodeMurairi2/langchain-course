#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
    timeout=30
)

hum_message = input("Enter your message here\n")

agent = create_agent(model=model,tools=[])

agent.invoke(hum_message)

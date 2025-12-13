#!/usr/bin/env python3

import os
from dotenv import load_dotenv
# import the required modules
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv("../model/auth.env")

# initialize the model
model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_AI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY")
    )

# create message objects
system_msg = SystemMessage("You are an helpful programming expert")
human_msg = HumanMessage("Tell me everything about FastAPI")

messages = [system_msg, human_msg]

# invoke the messages
response = model.invoke(messages)
print(response.content) # print AI messages
messages.append(response)
user_question = input("Ask your question...")
messages.append(user_question)
response = model.invoke(messages)
print(response.content)
print("__________________")
print(response.usage_metadata)

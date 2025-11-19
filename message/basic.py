#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv("../model/auth.env")

model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_AI_MODEL"),
                               api_key=os.getenv("GEMINI_API_KEY")
                               )

system_message = SystemMessage("You are a great education advisor")
human_message = HumanMessage("How to get scholarship?")

messages = [system_message, human_message]

second_messages = [
    {"role":"system", "content":"You are a great education advisor"},
    {"role":"user", "content":"How to get scholarship?"},
    {"role":"assistant", "content":"As your advisor, finding scholarship requires to start early."}
]
response = model.invoke(second_messages)
print(response.text)

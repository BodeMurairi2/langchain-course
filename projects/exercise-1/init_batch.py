#!/usr/bin/env python3

import os
import asyncio
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv("config.env")

model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_AI_MODEL_PRO"),
    api_key=os.getenv("GEMINI_API_PRO_KEY"),
    temperature=0.7,
    max_tokens=4000
    )

questions = [
    "List all countries in Africa",
    "Can you write a poem about love",
    "Can you send 5 motivational quotes"
]
response = model.batch_as_completed(questions)

for responses in response:
    print(f"AI:{responses[1].content}\n")

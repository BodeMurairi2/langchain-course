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
    temperature=0.1,
    max_tokens=1000
    )

question = f"""{input('Ask your question\n')}
            respond in plain text.
            No markdowns.
            No bullets points
            """

#for parts in model.stream(question):
#    print(parts.content.lstrip())

async def main():
    async for event in model.astream_events(question):
        if event["event"] == "on_chat_model_start":
            print(event["data"]["input"])
        
        elif event["event"] == "on_chat_model_stream":
            print(f"Token: {event['data']['chunk'].text}")
        
        elif event["event"] == "on_chat_model_end":
            print(f"Full message: {event['data']['output'].text}")

        else:
            pass

asyncio.run(main())

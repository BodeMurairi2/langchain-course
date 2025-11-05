#!/usr/bin/env python3
import sys
import io
import os
import time
import logging
from dotenv import load_dotenv

stderr_backup = sys.stderr
sys.stderr = io.StringIO()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

sys.stderr = stderr_backup

logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

load_dotenv()

def chat():
    """Virtual psychologist chat model"""
    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_AI_MODEL", "gemini-1.5-flash"),
        api_key=os.getenv("GEMINI_API_KEY")
    )

    history = [
        SystemMessage(content="""
        You are a compassionate and professional psychologist.
        Respond with empathy, emotional intelligence, and practical advice.
        Avoid diagnosing or prescribing medication.
        """),
        AIMessage(content="Hello! I’m your virtual psychologist assistant. How are you feeling today?")
    ]

    print("Welcome to your virtual psychologist assistant: \n")

    while True:
        try:
            question = input("You: ")
            if not question.strip():
                continue

            history.append(HumanMessage(content=question))
            response = model.invoke(history)
            time.sleep(2)
            print("\n")
            print(f"\nYour psychologist: {response.content}\n")
            history.append(AIMessage(content=response.content))

            again = input("Would you like to continue talking? (yes/no): ").lower()
            if again == "no":
                print("\nYour psychologist: I'm glad we talked today. Take care of yourself")
                break

        except KeyboardInterrupt:
            print("\nGoodbye")
            break

if __name__ == "__main__":
    chat()

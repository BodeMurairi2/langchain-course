#!/usr/bin/env python3

import os
import uuid
from dotenv import load_dotenv
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv("../model/auth.env")

model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_AI_MODEL"),
                               api_key=os.getenv("GEMINI_API_KEY"),
                               temperature=0.1,
                               max_tokens=4000
                               )

def start_chat():
    session_id = uuid.uuid4()
        
    sys_messages = SystemMessage(
    content=(
        "You are a helpful math tutor. "
        "Respond without using asterisks or markdown formatting. "
        "Provide detailed, clear, step-by-step explanations."
        )
        )
    ai_response = AIMessage(content="Hello! I am your virtual tutor. Ask me anything!\n")
    messages = [sys_messages, ai_response]
    
    print(ai_response.content)
    
    subject = input("Enter subject (math, physics, etc.): ").strip() or "general"
    level = input("Enter difficulty level (easy, medium, hard): ").strip() or "medium"
    
    while True:

        user_question = input("Ask your question here\n")
        
        if not user_question.strip():
            print("Enter a valid question")
            continue
        
        hum_messages = HumanMessage(
            content=user_question,
            metadata={
                "subject":subject,
                "level":level,
                "session_id":session_id,
                "timestamp":datetime.now().isoformat()

            }
            )
        
        messages.append(hum_messages) 

        try:
            ai_response = model.invoke(messages).text
        except Exception as error:
            print("An error occurred", error)
            continue

        print(ai_response)
        messages.append(AIMessage(content=ai_response))
        
        with open("chat_log.txt", "a") as f:
            log_entry = {
                "user": user_question,
                "ai": ai_response,
                "metadata": hum_messages.metadata
            }
            f.write(str(log_entry) + "\n")

        print("")
        
        computer_question = input("Enter y to continue and x to exit\n")
        if computer_question.lower() != "y":
            print("Thank you for using our model\n")
            break

if __name__ == "__main__":
    start_chat()

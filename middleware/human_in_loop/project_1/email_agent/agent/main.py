#!/usr/bin/env python3

import os
import uuid
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from tool.tool import send_email

def display_email(status: str, email_args: dict):
    print("\n" + "=" * 60)
    print(f" EMAIL STATUS: {status.upper()}")
    print("=" * 60)
    print(f"From   : {email_args['sender_email']}")
    print(f"To     : {email_args['receiver']}")
    print(f"Subject: {email_args['subject']}")
    print("-" * 60)
    print(email_args['body'])
    print("=" * 60 + "\n")

load_dotenv()

thread_id = str(uuid.uuid4())
memory = InMemorySaver()

# Create LLM (must support tools)
google_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_AI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY_V"),
    temperature=0.1,
    max_tokens=2000
)

# Tools
tools = [send_email]

# Create agent with Human-in-the-Loop middleware
agent = create_agent(
    model=google_llm,
    tools=tools,
    checkpointer=memory,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="Tool execution pending approval"
        )
    ]
)

SYS_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant.\n"
        "Draft emails responding to customer complaints.\n"
        "After drafting the email, you MUST call the send_email tool.\n"
        "Do not ask follow-up questions.\n"
        "Use ScholarVision as the Company's name in all the mails\n"
        "Use the provided information to fill all tool parameters."
    )
)

HUM_PROMPT = HumanMessage(
    content=(
        "Customer complaint: product arrives late and damaged.\n"
        "Sender email: bodemurairi2@gmail.com\n"
        "Receiver: b.murairi@alustudent.com\n"
        "Customer name: Bode Murairi\n"
        "Offer: replacement"
    )
)

response = agent.invoke(
    {"messages": [SYS_PROMPT, HUM_PROMPT]},
    config={"configurable": {"thread_id": thread_id}}
)

if "__interrupt__" in response:

    interrupt = response["__interrupt__"][0]
    action = interrupt.value["action_requests"][0]
    email_args = action["args"]

    # Show pending email
    display_email("PENDING APPROVAL", email_args)

    decision = input("Approve / Edit / Reject? ").strip().lower()

    if decision == "approve":
        result = agent.invoke(
            {"decision": "approve"},
            config={"configurable": {"thread_id": thread_id}}
        )

        print("\nTool execution result:")
        for msg in result["messages"]:
            if msg.type == "tool":
                print(msg.content)

    elif decision == "reject":
        agent.invoke(
            {"decision": "reject"},
            config={"configurable": {"thread_id": thread_id}}
        )

        display_email("REJECTED", email_args)

    elif decision == "edit":

        print("\nEnter new email body (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)

        new_body = "\n".join(lines)

        edited_args = {
            "subject": email_args["subject"],
            "body": new_body,
            "receiver": email_args["receiver"],
            "sender_email": email_args["sender_email"],
        }

        agent.invoke(
            {
                "decision": "edit",
                "edited_action": {
                    "name": "send_email",
                    "args": edited_args
                }
            },
            config={"configurable": {"thread_id": thread_id}}
        )

        display_email("EDITED & APPROVED", edited_args)

    else:
        print("Invalid decision. No action taken.")

else:
    print("No interrupt triggered. Tool was not called.")

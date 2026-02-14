#!/usr/bin/env python3

import os
from dotenv import load_dotenv
import textwrap

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

# For colored output in terminal
from colorama import init, Fore, Style
init(autoreset=True)

load_dotenv()

# Terminal width for wrapping messages
TERMINAL_WIDTH = 70

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_AI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY_V"),
    max_tokens=2000,
    temperature=0.5
)

agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model=llm,
            trigger=[("fraction", 0.4), ("messages", 10)],
            keep=("fraction", 0.2)
        )
    ]
)

sys_mes = SystemMessage(
    content="You are a love coach who gives powerful advices"
)

hum_mes = HumanMessage(
    content="My wife cheated on me. What should I do?"
)

response = agent.invoke(
    {"messages": [sys_mes, hum_mes]},
    config={"configurable": {"thread_id": "demo"}}
)

def print_chat_window(messages, width=TERMINAL_WIDTH):
    """
    Prints a chat-like interface in the terminal for System/Human/AI messages
    with proper text wrapping.
    """
    for msg in messages:
        if isinstance(msg, SystemMessage):
            prefix = Fore.CYAN + Style.BRIGHT + "[SYSTEM]: "
            color = Fore.CYAN
        elif isinstance(msg, HumanMessage):
            prefix = Fore.GREEN + Style.BRIGHT + "[YOU]: "
            color = Fore.GREEN
        elif isinstance(msg, AIMessage):
            prefix = Fore.MAGENTA + Style.BRIGHT + "[AI]: "
            color = Fore.MAGENTA
        else:
            prefix = "[UNKNOWN]: "
            color = Fore.WHITE

        # Wrap the message text
        wrapped_text = textwrap.fill(msg.content, width=width, subsequent_indent='    ')
        print(prefix + wrapped_text)
        print(Fore.WHITE + "-" * width)

print_chat_window(response["messages"])

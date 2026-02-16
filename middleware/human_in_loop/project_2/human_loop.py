#!/usr/bin/env python3

import os
import random
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, SummarizationMiddleware, ModelCallLimitMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.messages import HumanMessage, SystemMessage, AIMessage

from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

gemini_ai_config = {
    "gemini_model":os.getenv("GEMINI_AI_MODEL"),
    "gemini_model_api":os.getenv("GEMINI_API_KEY_V")
    }

if not all([gemini_ai_config["gemini_model"], gemini_ai_config["gemini_model_api"]]):
    raise ValueError(f"API Keys missing\nGEMINI MODEL: {gemini_ai_config['gemini_model']}\n GEMINI API KEY: {gemini_ai_config['gemini_model_api']}")

# memory
memory_config = {
    "memory":InMemorySaver(),
    "thread_id":random.randrange(1,999999999)
}

llm = ChatGoogleGenerativeAI(
    model = gemini_ai_config["gemini_model"],
    api_key = gemini_ai_config["gemini_model_api"],
    temperature=0.5,
    max_tokens=1500
    )


# agent
agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=memory_config["memory"],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "tool_name": {
                    "allowed_decisions":["approve","edit", "reject"]
                }
            }
        ),
        SummarizationMiddleware(
            model=llm,
            trigger=[("messages",5),("fraction",0.3)],
            keep=("message",5)
        ),
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="error",
            error_message="Number of requests exceeded. Try later"
        )
    ],
    system_prompt="You are a powerful customers assistant agent"
)

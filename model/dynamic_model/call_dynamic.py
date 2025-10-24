#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

load_dotenv("../auth.env")

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise FileNotFoundError("No API KEY")

os.environ["GOOGLE_API_KEY"] = API_KEY

basic_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
advanced_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

@wrap_model_call
def switch_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Dynamic model selection based on conversation length"""
    message_count = len(request.state["messages"])
    if message_count > 15:
        request.model = advanced_model
    else:
        request.model = basic_model
    return handler(request)

agent = create_agent(
    model=basic_model,
    middleware=[switch_model_selection]
)

question = "Explain NLP and embeddings in simple terms."
result = agent.invoke([{"role": "user", "content": question}])
print(result["text"])

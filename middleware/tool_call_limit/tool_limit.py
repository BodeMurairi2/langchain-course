#!/usr/bin/env python3

import os
import random
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents.middleware import ToolCallLimitMiddleware

load_dotenv()

gemini_config = {
    "gemini_model":os.getenv(""),
    "gemini_api_key":os.getenv("")
}

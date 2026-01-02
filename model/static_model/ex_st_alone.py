#!/usr/bin/env python3

from langchain.chat_models import init_chat_model

model = init_chat_model("google_genai:gemini-2.5-flash-lite")
print(model.invoke("Hello bro"))

#!/usr/bin/env python3

from langchain.agents import AgentState

class CustomAgentstate(AgentState):
    user_id:str
    preferences:dict


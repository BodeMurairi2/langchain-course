#!/usr/bin/env python3

from langchain.tools import tool

@tool
def search(query:str)->str:
    """search information"""
    return f"Result for :{query}"

@tool
def get_weather(location:str)->str:
    """Get weather data for a particular location"""
    return f"Weather in location: {location}"

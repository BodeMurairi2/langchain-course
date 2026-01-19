#!/usr/bin/env python3

import requests

import os
from dotenv import load_dotenv
from pathlib import Path

from pydantic import BaseModel, Field
from typing import Literal, Any

from langchain.tools import tool, ToolRuntime
from langchain.messages import RemoveMessage
from langchain.agents import AgentState
from langchain.agents.middleware import before_model

from langgraph.runtime import Runtime
from langgraph.graph.message import REMOVE_ALL_MESSAGES

env_path = Path(__file__).parent.parent/"weather.env"
load_dotenv(env_path)

class Weather(BaseModel):
    """Pydantic for weather api class"""
    location:str=Field(description="Location type city where you want to get the weather report")
    api:str=Field(default="yes",
                   description="Choose ye to get air quality in the weather result, no to remove air quality info"
                   )
    
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=Weather)
def get_instant_weather(location:str,
                        api:str,
                        include_forecast:bool
                        ):
    f"""
    This tool gets instant weather information
    from city {location}
    Args:
        location:str = Field(Description='Name of the city')
    """
    
    weather_api_key = os.getenv("WEATHER_API_KEY")
    if not weather_api_key:
        raise ValueError("Weather API key is missing")
    
    parameter= {
        "key":weather_api_key
    }

    base_url = "http://api.weatherapi.com/v1"

    user_data = Weather(location=location)
    location = user_data.location
    current_weather_url = f"{base_url}/current.json?q={location}"
    
    try:
        weather_data = requests.get(url=current_weather_url,params=parameter, timeout=30)
        weather_data.raise_for_status()
    except Exception as error:
        return {"message":"Oops! An error occured","error":error}
    
    result = weather_data.json()
    return {
        "location":result["location"]["name"],
        "localtime":result["location"]["localtime"],
        "temperature_c":result["current"]["temp_c"],
        "condition":result["current"]["condition"],
        "wind_mph":result["current"]["wind_mph"],
        "wind_kph":result["current"]["wind_kph"],
        "wind_degree":result["current"]["wind_degree"],
        "wind_dir":result["current"]["wind_dir"],
        "pressure_mb":result["current"]["pressure_mb"],
        "pressure_in":result["current"]["pressure_in"],
        "precip_mm":result["current"]["precip_mm"],
        "precip_in":result["current"]["precip_in"],
        "humidity":result["current"]["humidity"],
        "cloud":result["current"]["cloud"],
        "feelslike_c":result["current"]["feelslike_c"]
        }

@before_model
def trim_messages(state:AgentState, runtime:Runtime):
    """Keep the last messages four messages"""
    messages = state["messages"]

    if len(messages) <= 3:
        return None
    
    first_messages = messages[0]
    recent_messages = messages[-4:] if len(messages) < 10 else messages[-3:]
    new_messages = [first_messages] + recent_messages
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

if __name__ == "__main__":
    city = "Kigali"
    result = get_instant_weather.invoke({
        "location":city,
        "api":"yes",
        "include_forecast":False
    })
    print(result)

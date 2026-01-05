#!/usr/bin/env python3

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

# define using pydantic schema
class WeatherInput(BaseModel):
    """Input for weather query"""
    location:str = Field(..., description="City name or location")
    units:Literal["celsius", "fahrenheit"] = Field(..., default="celsius",
                                                   description="Temperature unit preference"
                                                   )
    include_forecast:bool = Field(...,default=False, description="Include x days forecast")
    
# define using json schema
weather_input = {
    "type":"object",
    "properties":{
        "location":{"type":"string"},
        "units":{"type":"string"},
        "include_forecast":{"type":"boolean"}
    },
    "required":["location", "units", "include_forecast"]
}

@tool(args_schema=WeatherInput)
def get_weather(location:str, units:str, include_forecast:bool):
    """
    Get current weather and optional forecast
    Docstring for get_weather
    
    :param location: Description
    :type location: str
    :param units: Description
    :type units: str
    :param include_forecast: Description
    :type include_forecast: bool
    """
    temp = 22 if units == "celcius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result +="\nNext five days: sunny"
    return result

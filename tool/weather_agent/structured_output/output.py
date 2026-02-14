#!/usr/bin/env python3

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Response_output(BaseModel):
    """This class provides the structure output"""
    location:str = Field(description="The city name for which the weather information is provided")
    location_time:datetime = Field(description="The local time at the specified location")
    temperature_celsius:str = Field(description="The current temperature in Celsius at the specified location")
    pressure_mb:str = Field(description="The current atmospheric pressure in millibars at the specified location")
    precip_mm:str = Field(description="The current precipitation in millimeters at the specified location")
    humidity:str = Field(description="The current humidity percentage at the specified location")
    feellike_c:str = Field(description="The current 'feels like' temperature in Celsius at the specified location")
    last_City:str = Field(description="The last city for which the weather information was provided")
    ai_response:str = Field(description="The AI's response to the user's query, including any advice or recommendations based on the weather data")


"""use a Json schema for structured output to ensure the response is consistent and can be easily parsed by other systems or tools."""
structured_output = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "location_time": {"type": "string"},
        "temperature_celsius": {"type": "number"},
        "pressure_mb": {"type": "number"},
        "precip_mm": {"type": "number"},
        "humidity": {"type": "number"},
        "feellike_c": {"type": "number"},
        "last_City": {"type": "string"},
        "ai_response": {"type": "string"}
    },
    "required": [
        "location",
        "location_time",
        "temperature_celsius",
        "pressure_mb",
        "precip_mm",
        "humidity",
        "feellike_c",
        "last_City",
        "ai_response"
    ]
}

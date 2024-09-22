import json

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from simple_ai_agents.chat_session import ChatLLMSession
from simple_ai_agents.models import Tool

load_dotenv()


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
        )
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


weather_tool_schema = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}


class Weather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(description="Location of the weather")
    unit: str = Field(description="Unit of the temperature")


@pytest.mark.parametrize(
    "tools",
    [
        [Tool(tool_model=weather_tool_schema, function=get_current_weather)],
        [Tool(tool_model=Weather, function=get_current_weather)],
        [Tool(function=get_current_weather)],
    ],
)
def test_gen_with_tool(tools):
    sess = ChatLLMSession()
    response = sess.gen(
        "What's the weather like in San Francisco",
        tools=tools,
    )
    assert "72" in response


@pytest.mark.parametrize(
    "tools",
    [
        [Tool(tool_model=weather_tool_schema, function=get_current_weather)],
        [Tool(tool_model=Weather, function=get_current_weather)],
        [Tool(function=get_current_weather)],
    ],
)
@pytest.mark.asyncio
async def test_gen_model_async(tools):
    sess = ChatLLMSession()
    response = await sess.gen_async(
        "What's the weather like in San Francisco",
        tools=tools,
    )
    assert "72" in response

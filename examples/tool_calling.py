import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from simple_ai_agents.chat_session import ChatLLMSession
from simple_ai_agents.models import LLMOptions, Tool

load_dotenv()

openai = LLMOptions(model="gpt-4o-mini", temperature=0.7)
github = LLMOptions(model="github/gpt-4o-mini")
groq = LLMOptions(model="groq/llama-3.1-70b-versatile", temperature=0.7)
together = LLMOptions(
    model="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", temperature=0.7
)
anthropic = LLMOptions(model="claude-3-5-sonnet-20240620")
llama3 = LLMOptions(model="ollama/llama3.1", api_base="http://localhost:11434")
cerebras = LLMOptions(model="cerebras/llama3.1-70b")


def get_current_weather(location, unit="fahrenheit"):
    """
    Get the current weather in a given location

    Args:
        location (str): The city and state, e.g. San Francisco, CA
        unit (str): The unit of the temperature, either celsius or fahrenheit
    """
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
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


def run():
    sess = ChatLLMSession()
    response = sess.gen(
        "What's the weather like in San Francisco",
        system="Use the tools provided to answer the question",
        llm_options=github,
        tools=[
            # Tool(tool_model=weather_tool_schema, function=get_current_weather),
            # Tool(tool_model=Weather, function=get_current_weather),
            Tool(function=get_current_weather),
        ],
        tool_choice="auto",
    )
    print(response)
    # Tool calling also works with stream method
    # stream = sess.stream(
    #     "What's the weather like in San Francisco",
    #     llm_options=github,
    #     tools=[
    #         Tool(tool_model=weather_tool_schema, function=get_current_weather),
    #     ],
    #     tool_choice="auto",
    # )
    # for chunk in stream:
    #     print(chunk["delta"])


if __name__ == "__main__":
    run()

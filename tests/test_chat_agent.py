import json
import logging
import os

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from simple_ai_agents.chat_agent import ChatAgent, ChatAgentAsync
from simple_ai_agents.models import Tool

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
load_dotenv()


class UserDetail(BaseModel):
    name: str
    age: int


def test_new_session():
    chatbot = ChatAgent()
    session = chatbot.new_session(return_session=True)
    assert session in chatbot.sessions.values()


def test_get_session():
    chatbot = ChatAgent()
    session = chatbot.new_session()
    assert chatbot.get_session(session.id) == session


def test_list_sessions():
    chatbot = ChatAgent()
    session1 = chatbot.get_session()
    session2 = chatbot.new_session()
    assert set(chatbot.list_sessions()) == set({session1.id, session2.id})


def test_save_session():
    output_path_csv = "./tests/output/chat_session.csv"
    output_path_json = "./tests/output/chat_session.json"
    if os.path.exists(output_path_csv):
        os.remove(output_path_csv)
    if os.path.exists(output_path_json):
        os.remove(output_path_json)
    chatbot = ChatAgent()
    session = chatbot.get_session()
    chatbot("Hello")
    chatbot.save_session(output_path_csv, session.id)
    assert os.path.exists(output_path_csv)
    chatbot.save_session(output_path_json, session.id, format="json")
    assert os.path.exists(output_path_json)


def test_load_session_csv():
    input_path_csv = "./tests/input/chat_session.csv"
    chatbot = ChatAgent()
    session = chatbot.load_session(input_path_csv)
    assert len(session.messages) == 2  # type: ignore
    assert session.messages[0].content == "Hello"  # type: ignore


def test_load_session_json():
    input_path_json = "./tests/input/chat_session.json"
    chatbot = ChatAgent()
    session = chatbot.load_session(input_path_json)
    assert len(session.messages) == 2  # type: ignore
    assert session.messages[0].content == "Hello"  # type: ignore


def test_reset_session():
    chatbot = ChatAgent()
    session = chatbot.get_session()
    _ = chatbot("Hello")
    chatbot.reset_session(session.id)
    assert session.messages == []


def test_delete_session():
    chatbot = ChatAgent()
    session = chatbot.new_session()
    chatbot.delete_session(session.id)
    assert session.id not in chatbot.sessions


def test_call():
    chatbot = ChatAgent()
    response = chatbot("Hello")
    assert isinstance(response, str)


def test_default_build_system():
    chatbot = ChatAgent()
    assert chatbot.build_system() == "You are a helpful assistant."


def test_gen_model():
    chatbot = ChatAgent()
    response = chatbot.gen_model("Generate a user", response_model=UserDetail)
    assert response.name is not None
    assert response.age is not None


@pytest.mark.asyncio
async def test_chat_agent_async():
    chatbot = ChatAgentAsync()
    response = await chatbot("Hello")
    assert isinstance(response, str)


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


def test_chat_agent_with_tools():
    chatbot = ChatAgent(tools=[Tool(function=get_current_weather)])
    response = chatbot("Hi, what is the weather in San Francisco?")
    assert isinstance(response, str)
    assert "72" in response

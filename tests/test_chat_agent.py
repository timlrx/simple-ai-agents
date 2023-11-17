import logging

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from simple_ai_agents.chat_agent import ChatAgent, ChatAgentAsync

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

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from simple_ai_agents.chat_session import ChatLLMSession
from simple_ai_agents.models import LLMOptions

load_dotenv()


class UserDetail(BaseModel):
    name: str
    age: int


def test_prepare_request():
    sess = ChatLLMSession()
    prompt = "Hello, how can I help you?"
    system = "Test system"
    llm_options: LLMOptions = {"model": "openai/gpt-4o-mini", "temperature": 0.5}
    model, kwargs, history, user_message, llm_provider, _, _ = sess.prepare_request(
        prompt, system=system, llm_options=llm_options
    )
    assert model == "openai/gpt-4o-mini"
    assert kwargs == {"temperature": 0.5}
    assert history == [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    assert user_message.role == "user"
    assert user_message.content == prompt
    assert llm_provider == "openai"


def test_prepare_model_request_openai():
    sess = ChatLLMSession()
    prompt = "Parse this input"

    llm_options: LLMOptions = {"model": "openai/gpt-4o-mini", "temperature": 0.5}
    (
        model,
        kwargs,
        history,
        user_message,
        llm_provider,
        response_model,
        mode,
    ) = sess.prepare_request(prompt, llm_options=llm_options, response_model=UserDetail)
    assert model == "openai/gpt-4o-mini"
    assert "temperature" in kwargs
    assert "tool_choice" in kwargs
    assert history == [
        {"role": "user", "content": prompt},
    ]
    assert user_message.role == "user"
    assert user_message.content == prompt
    assert llm_provider == "openai"
    assert response_model.__name__ == "UserDetail"  # type: ignore


def test_prepare_model_request_ollama():
    sess = ChatLLMSession()
    prompt = "Parse this input"

    llm_options: LLMOptions = {"model": "ollama/mistral", "temperature": 0.5}
    (
        model,
        kwargs,
        history,
        user_message,
        llm_provider,
        response_model,
        mode,
    ) = sess.prepare_request(prompt, llm_options=llm_options, response_model=UserDetail)
    assert model == "ollama/mistral"
    assert "temperature" in kwargs
    assert "tool_choice" in kwargs
    assert user_message.role == "user"
    assert user_message.content == prompt
    assert llm_provider == "ollama"
    assert response_model.__name__ == "UserDetail"  # type: ignore


def test_gen_model():
    sess = ChatLLMSession()
    response = sess.gen_model("Generate a user", response_model=UserDetail)
    assert response.name is not None
    assert response.age is not None


@pytest.mark.asyncio
async def test_gen_model_async():
    sess = ChatLLMSession()
    response = await sess.gen_model_async("Generate a user", response_model=UserDetail)
    assert response.name is not None
    assert response.age is not None


def test_gen():
    sess = ChatLLMSession()
    prompt = "1+1"
    response = sess.gen(prompt)
    assert len(response) > 0
    assert sess.total_prompt_length > 0
    assert sess.total_completion_length > 0
    assert sess.total_length > 0


@pytest.mark.asyncio
async def test_gen_async():
    sess = ChatLLMSession()
    prompt = "1+1"
    response = await sess.gen_async(prompt)
    assert len(response) > 0
    assert sess.total_prompt_length > 0
    assert sess.total_completion_length > 0
    assert sess.total_length > 0


def test_stream():
    sess = ChatLLMSession()
    prompt = "1+1"
    response = sess.stream(prompt)
    for i in response:
        assert len(i["delta"]) > 0
        assert len(i["response"]) > 0


@pytest.mark.asyncio
async def test_stream_async():
    sess = ChatLLMSession()
    prompt = "1+1"
    response = sess.stream_async(prompt)
    async for i in response:
        assert len(i["delta"]) > 0
        assert len(i["response"]) > 0


def test_stream_model():
    sess = ChatLLMSession()
    response = sess.stream_model("Generate a user", response_model=UserDetail)
    for obj in response:
        assert "name" in obj.model_fields.keys()
        assert "age" in obj.model_fields.keys()


@pytest.mark.asyncio
async def test_stream_model_async():
    sess = ChatLLMSession()
    response = sess.stream_model_async("Generate a user", response_model=UserDetail)
    async for obj in response:
        assert "name" in obj.model_fields.keys()
        assert "age" in obj.model_fields.keys()

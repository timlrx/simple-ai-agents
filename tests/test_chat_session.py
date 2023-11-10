import pytest
from dotenv import load_dotenv

from simple_ai_agents.chat_session import ChatLLMSession
from simple_ai_agents.models import LLMOptions

load_dotenv()


def test_prepare_request():
    ai = ChatLLMSession()
    prompt = "Hello, how can I help you?"
    system = "Test system"
    llm_options: LLMOptions = {"model": "test-model", "temperature": 0.5}
    model, other_options, history, user_message = ai.prepare_request(
        prompt, system=system, llm_options=llm_options
    )
    assert model == "test-model"
    assert other_options == {"temperature": 0.5}
    assert history == [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    assert user_message.role == "user"
    assert user_message.content == prompt


def test_gen():
    ai = ChatLLMSession()
    prompt = "1+1"
    response = ai.gen(prompt)
    assert len(response) > 0
    assert ai.total_prompt_length > 0
    assert ai.total_completion_length > 0
    assert ai.total_length > 0


@pytest.mark.asyncio
async def test_gen_async():
    ai = ChatLLMSession()
    prompt = "1+1"
    response = await ai.gen_async(prompt)
    assert len(response) > 0
    assert ai.total_prompt_length > 0
    assert ai.total_completion_length > 0
    assert ai.total_length > 0


def test_stream():
    ai = ChatLLMSession()
    prompt = "1+1"
    response = ai.stream(prompt)
    for i in response:
        assert len(i["delta"]) > 0
        assert len(i["response"]) > 0


@pytest.mark.asyncio
async def test_stream_async():
    ai = ChatLLMSession()
    prompt = "1+1"
    response = ai.stream_async(prompt)
    async for i in response:
        assert len(i["delta"]) > 0
        assert len(i["response"]) > 0

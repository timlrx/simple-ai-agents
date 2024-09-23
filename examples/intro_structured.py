import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich import print
from rich.console import Console

from simple_ai_agents.chat_session import ChatLLMSession
from simple_ai_agents.models import LLMOptions

load_dotenv()

openai = LLMOptions(model="gpt-4o-mini", temperature=0.7)
anthropic = LLMOptions(model="claude-3-5-sonnet-20240620")
groq = LLMOptions(model="groq/llama-3.1-8b-instant")
github = LLMOptions(model="github/gpt-4o-mini")
anyscale = LLMOptions(
    model="anyscale/mistralai/Mistral-7B-Instruct-v0.1", temperature=0.7
)
together = LLMOptions(
    model="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", temperature=0.7
)
mistral = LLMOptions(
    model="ollama/mistral", temperature=0.7, api_base="http://localhost:11434"
)
llama3 = LLMOptions(model="ollama/llama3.1", api_base="http://localhost:11434")
cerebras = LLMOptions(model="cerebras/llama3.1-70b")


class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")


console = Console()
sess = ChatLLMSession(llm_options=github)


def parse_gen_model(text):
    parsed = sess.gen_model(
        text,
        response_model=Person,
    )
    print("Gen Model Sync:", parsed)
    return parsed


def parse_stream_model(text):
    parsed = sess.stream_model(
        text,
        response_model=Person,
    )
    print("Streaming Model Sync...")
    for response in parsed:
        print(response)
    print("Streaming Model Sync Done")
    return parsed


async def parse_gen_model_async(text):
    parsed = await sess.gen_model_async(
        text,
        response_model=Person,
    )
    print("Gen Model Async:", parsed)
    return parsed


async def parse_stream_model_async(text):
    parsed = sess.stream_model_async(
        text,
        response_model=Person,
    )
    print("Streaming Model Async...")
    async for response in parsed:
        print(response)
    print("Streaming Model Async Done")
    return parsed


if __name__ == "__main__":
    # Since parsed is a pydantic model, we can easily add it to a database or serve it as an API
    text = "Extract `My name is John and I am 18 years old` into JSON"
    parsed = parse_gen_model(text)
    _ = parse_stream_model(text)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(parse_gen_model_async(text))
    loop.run_until_complete(parse_stream_model_async(text))

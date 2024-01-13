from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich import print

from simple_ai_agents.chat_agent import ChatAgent
from simple_ai_agents.models import LLMOptions

load_dotenv()

openai: LLMOptions = {"model": "gpt-3.5-turbo", "temperature": 0.7}
anyscale: LLMOptions = {
    "model": "anyscale/mistralai/Mistral-7B-Instruct-v0.1",
    "temperature": 0.7,
}
together: LLMOptions = {
    "model": "together_ai/togethercomputer/CodeLlama-34b",
    "temperature": 0.7,
}
mistral: LLMOptions = {
    "model": "ollama/mistral",
    "temperature": 0.7,
    "api_base": "http://localhost:11434",
}


class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")


def parse(text):
    chatbot = ChatAgent(llm_options=anyscale)
    parsed = chatbot.gen_model(
        text,
        response_model=Person,
    )
    print(parsed)
    return parsed


if __name__ == "__main__":
    parsed = parse("Extract `My name is John and I am 18 years old` into JSON")
    # Since parsed is a pydantic model, we can easily add it to a database or serve it as an API

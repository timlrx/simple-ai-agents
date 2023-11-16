import logging

from dotenv import load_dotenv
from rich import print

from simple_ai_agents.chat_agent import ChatAgent
from simple_ai_agents.models import LLMOptions

logging.basicConfig(level=logging.INFO, format="%(message)s")
load_dotenv()

openai: LLMOptions = {"model": "gpt-3.5-turbo", "temperature": 0.7}
llama: LLMOptions = {
    "model": "huggingface/meta-llama/Llama-2-13b-chat-hf",
    "temperature": 0.5,
}
mistral: LLMOptions = {
    "model": "ollama/mistral",
    "temperature": 0.7,
    "api_base": "http://localhost:11434",
}

default_options = openai

if __name__ == "__main__":
    chatbot = ChatAgent(
        system="You are a helpful assistant", console=False, llm_options=default_options
    )
    results = chatbot("Generate 2 random numbers between 0 to 100")
    print(results)
    results = chatbot("Which of the two numbers is bigger?")
    print(results)
    chatbot.new_session()
    print("new session created...")
    results = chatbot("Which of the two numbers is bigger?")
    print(results)

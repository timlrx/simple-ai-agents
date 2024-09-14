from dotenv import load_dotenv

from simple_ai_agents.chat_agent import ChatAgent
from simple_ai_agents.models import LLMOptions

# logging.basicConfig(level=logging.INFO, format="%(message)s")
load_dotenv()

openai: LLMOptions = {"model": "gpt-4o-mini", "temperature": 0.7}
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
        system="You are a helpful assistant", llm_options=default_options
    )
    results = chatbot("Generate 2 random numbers between 0 to 100", console_output=True)
    results = chatbot("Which of the two numbers is bigger?", console_output=True)
    chatbot.new_session()
    results = chatbot("Which of the two numbers is bigger?", console_output=True)

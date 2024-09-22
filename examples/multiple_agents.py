from dotenv import load_dotenv
from rich import print

from simple_ai_agents.chat_agent import ChatAgent
from simple_ai_agents.models import LLMOptions

load_dotenv()

mistral: LLMOptions = {
    "model": "ollama/mistral",
    "temperature": 0.7,
    "api_base": "http://localhost:11434",
}

llama3: LLMOptions = {
    "model": "ollama/llama3.1",
    "temperature": 0.7,
    "api_base": "http://localhost:11434",
}


def store_conversation():
    """
    Simulate a conversation between a customer and a storekeeper.
    Both agents use different ollama models.
    """
    storekeeper = ChatAgent(
        system="You are a helpful storekeeper",
        character="storekeeper",
        llm_options=mistral,
        ai_text_color="bright_blue",
    )
    customer = ChatAgent(
        system="You are a customer at a shop. Speak concisely and clearly.",
        character="customer",
        llm_options=llama3,
        ai_text_color="bright_green",
    )
    runs = 0
    customer_reply = "Hi"
    while runs < 3:
        storekeeper_reply = storekeeper(customer_reply, console_output=True)
        customer_reply = customer(storekeeper_reply, console_output=True)
        runs += 1
    print("---")
    print(storekeeper.get_session())


if __name__ == "__main__":
    store_conversation()

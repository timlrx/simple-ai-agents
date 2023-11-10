import logging

from dotenv import load_dotenv
from rich import print

from simple_ai_agents.chat_agent import ChatAgent

logging.basicConfig(level=logging.INFO, format="%(message)s")
load_dotenv()

chatbot = ChatAgent(system="You are a helpful assistant", console=False)
results = chatbot("Generate 2 random numbers between 0 to 100")
print(results)
results = chatbot("Which of the two numbers is bigger?")
print(results)
chatbot.new_session()
print("new session created...")
results = chatbot("Which of the two numbers is bigger?")
print(results)

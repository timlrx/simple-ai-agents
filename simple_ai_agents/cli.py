import argparse

import fire
from dotenv import load_dotenv

from simple_ai_agents.chat_agent import ChatAgent
from simple_ai_agents.prompts import SYSTEM_PROMPT

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("character", help="Specify the character", default=None, nargs="?")
parser.add_argument("--prime", action="store_true", help="Enable priming")

ARGS = parser.parse_args()


def interactive_chat():
    ChatAgent(
        character=ARGS.character,
        prime=ARGS.prime,
        console=True,
        system=SYSTEM_PROMPT,
    )


if __name__ == "__main__":
    fire.Fire(interactive_chat)

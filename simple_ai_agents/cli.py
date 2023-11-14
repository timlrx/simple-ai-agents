import click
from dotenv import load_dotenv

from simple_ai_agents.chat_agent import ChatAgent
from simple_ai_agents.prompts import SYSTEM_PROMPT

load_dotenv()


@click.command()
@click.option("--character", default=None, help="Specify the character")
@click.option("--prime/--no-prime", default=False, help="Enable priming")
def interactive_chat(character, prime):
    ChatAgent(
        character=character,
        prime=prime,
        console=True,
        system=SYSTEM_PROMPT,
    )


if __name__ == "__main__":
    interactive_chat()

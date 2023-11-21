import click
from dotenv import load_dotenv

from simple_ai_agents.chat_agent import ChatAgent
from simple_ai_agents.prompts import SYSTEM_PROMPT

load_dotenv()


@click.command()
@click.option("--character", default=None, help="Specify the character")
@click.option("--prime/--no-prime", default=False, help="Enable priming")
@click.option(
    "--model",
    default="gpt-3.5-turbo",
    help="""Specify the LLM model e.g. gpt-3.5-turbo, ollama/mistral.
    Uses gpt-3.5-turbo by default.""",
)
@click.option("--temperature", default=0.7, help="LLM temperature. Default is 0.7.")
def interactive_chat(character, prime, model, temperature):
    ChatAgent(
        character=character,
        prime=prime,
        console=True,
        system=SYSTEM_PROMPT,
        llm_options={"model": model, "temperature": temperature},
    )


if __name__ == "__main__":
    interactive_chat()

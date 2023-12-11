import sys

import click
from dotenv import load_dotenv

from simple_ai_agents.chat_agent import ChatAgent
from simple_ai_agents.prompts import SYSTEM_PROMPT

load_dotenv()


@click.command()
@click.option("--character", default=None, help="Specify the character")
@click.option("--prime/--no-prime", default=False, help="Enable priming")
@click.option(
    "-m",
    "--model",
    default="gpt-3.5-turbo",
    help="""Specify the LLM model e.g. gpt-3.5-turbo, ollama/mistral.
    Uses gpt-3.5-turbo by default.""",
)
@click.option("--temperature", default=0.7, help="LLM temperature. Default is 0.7.")
@click.option("-s", "--system", default=SYSTEM_PROMPT, help="System prompt")
@click.option(
    "--display-names/--no-display-names", default=False, help="Display character names"
)
@click.argument("prompt", required=False)
def interactive_chat(
    character, prime, model, temperature, system, display_names, prompt
):
    def read_prompt():
        """Read prompt from stdin if available"""
        nonlocal prompt
        stdin_prompt = None
        if not sys.stdin.isatty():
            stdin_prompt = click.get_text_stream("stdin").read().strip()
        if stdin_prompt:
            bits = [stdin_prompt]
            if prompt:
                bits.append(prompt)
            prompt = " ".join(bits)
            # https://stackoverflow.com/questions/46129898/conflict-between-sys-stdin-and-input-eoferror-eof-when-reading-a-line
            sys.stdin = open("/dev/tty")
        return prompt

    prompt = read_prompt()
    ChatAgent(
        character=character,
        prime=prime,
        console=True,
        system=system,
        prompt=prompt,
        display_names=display_names,
        llm_options={"model": model, "temperature": temperature},
    )


if __name__ == "__main__":
    interactive_chat()

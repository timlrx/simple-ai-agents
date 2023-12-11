## Simple AI Agents

Create simple multi-agent workflows using any LLMs - easy to experiment, use and deploy.

The package extends [Simple AI Chat][simple-ai-chat] by adding support for [100+ LLM providers][litellm], [structured responses][instructor] and multiple agents, similar to [Autogen][autogen]. With out of the box handling of LLM requests, session handling and structured response generation, multi-agent conversations can be easily orchestrated with Python to manage the control flow. This results in code which is easy to understand and extend with minimal dependencies.

__Note__: The package is in active development and the API is subject to change. To use, please clone the repo and install the package locally. It has not been published to PyPI yet. Better JSON support for other LLM models and the ability to save and load sessions are in the roadmap.

## Features

- Mix and match LLM providers ([OpenAI, Huggingface, Ollama, Anthropic and more!][litellm]).
- Create and run chats with only a few lines of code!
- Integrates with [instructor][instructor] to provide structured responses.
- Run multiple independent chats at once or create [Autogen][autogen] like multi-agent conversations.
- Minimal codebase: no code dives to figure out what's going on under the hood needed!
- Async and streaming.
- Interactive CLI.

## Getting Started

Add the necessary environment variables corresponding to the [LLM providers](https://docs.litellm.ai/docs/providers) that will be used. Refer to `.env.example` for an example of the variables to add:

```
OPENAI_API_KEY=<your openai api key>
OPENAI_ORGANIZATION=<your openai organization>
HUGGINGFACE_API_KEY=<your huggingfacehub api token>
```

The package provides a `ChatAgent` class that can be used to create a chatbot:

```py
from simple_ai_agents.chat_agent import ChatAgent

chatbot = ChatAgent(system="You are a helpful assistant")
chatbot("Generate 2 random numbers between 0 to 100", console_output=True)
chatbot("Which of the two numbers is bigger?", console_output=True)
```

`console_output` provides a convenient way to print the chatbot's response to the console. By default, the chatbot uses the `openai` provider. To use a different provider, pass the `llm_options` argument to the `ChatAgent` constructor. For example, to use the mistral model from [ollama][ollama]:

```py
from simple_ai_agents.models import LLMOptions

mistral: LLMOptions = {
    "model": "ollama/mistral",
    "temperature": 0.7,
    "api_base": "http://localhost:11434",
}
chatbot = ChatAgent(system="You are a helpful assistant", llm_options=mistral)
```

The CLI offers an easy way to start a local chatbot session similar to [Simple AI Chat][simple-ai-chat] or [Ollama][ollama] but with support for almost all LLM providers.

See the examples folder for other use cases.

### CLI

Ensure that you have the necessary environment variables set up. Usage:

```sh
aichat [OPTIONS] [PROMPT]
```

The CLI supports the following options:
- `--prime`: Prime the chatbot with a prompt before starting the chat.
- `--character`: The name of the chat agent.
- `--model`: Specify the LLM model e.g. gpt-3.5-turbo, ollama/mistral etc. Uses gpt-3.5-turbo by default.
- `--temperature`: Specify the temperature for the LLM model. Uses 0.7 by default.
- `--system`: System prompt.
- `--help`

#### Interactive open-ended chat

```sh
aichat --prime
```

#### Pass in prompts as arguments

Uses a local instance of the mistral model from [ollama][ollama] to summarize the README file.

```sh
cat README.md | aichat --model ollama/mistral "Summarize this file"
```

Looking for an option that is not available? Open an issue or submit a PR!

### Examples

- [Basic chatbot session](examples/sessions.py)
- [Multiple chatbot sessions](examples/chatbot_session.py)
- [Multi-agent conversation with different models](examples/multiple_agents.py)
- [Structured responses](examples/structured_responses.py)

## Development

### Poetry

Package management is handled by `poetry`. Install it by following the instructions [here](https://python-poetry.org/docs/#installation).

### Installing packages

After installing `poetry`, install the project packages by running:

```bash
poetry install
```

### Setting up pre-commit hooks

Pre-commit hooks automatically process your code prior to commit, enforcing consistency across the codebase without the need for manual intervention. Currently, these hooks are used:

- `trailing-whitespace`: trims trailing whitespace
- `requirements-txt-fixer`: sorts entries in requirements.txt
- `black`: to format the code consistently
- `isort`: to sort all imports consistently
- `flake8`: as a linter

All commits should pass the hooks above before being pushed.

```bash
# Install the configured hooks
poetry run pre-commit install

# Run the hooks on all files in the repo
poetry run pre-commit run -a
```

If there are any failures, resolve them first, then stage and commit as usual. Committing will automatically trigger these hooks, and the commit will fail if there are unresolved errors.

### Run the tests

```bash
poetry run pytest
```

[simple-ai-chat]: https://github.com/minimaxir/simpleaichat
[litellm]: https://litellm.vercel.app/docs/providers
[instructor]: https://github.com/jxnl/instructor
[autogen]: https://github.com/microsoft/autogen
[ollama]: https://ollama.ai/
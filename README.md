## Simple AI Agents

AI Agents inspired by [Simple AI Chat][simple-ai-chat] - easy to experiment, easy to use, easy to deploy.

Simple ai agents is a Python package for easily interfacing with chat apps like ChatGPT and GPT-4 with robust features and minimal code complexity. This tool has many features optimized for working with ChatGPT as fast and as cheap as possible, but still much more capable of modern AI tricks than most implementations:

- Mix and match LLM providers ([OpenAI, Huggingface, Ollama, Anthropic and more!][litellm]).
- Create and run chats with only a few lines of code!
- Optimized workflows which minimize the amount of tokens used, reducing costs and latency.
- Run multiple independent chats at once.
- Minimal codebase: no code dives to figure out what's going on under the hood needed!
- Chat streaming responses and the ability to use tools.
- Async and streaming.
- Up next: tools and function calling

## Getting Started

Add the necessary environment variables corresponding to the LLM providers that will be used. Refer to `.env.example` for an example of the variables to add:

```
OPENAI_API_KEY=<your openai api key>
OPENAI_ORGANIZATION=<your openai organization>
HUGGINGFACEHUB_API_TOKEN=<your huggingfacehub api token>
```

### CLI

Run the chatbot in the terminal with:

```bash
python simple_ai_agents/cli.py --prime
```

### Example

Run the `demo.py` script to see an example of how the chatbot works.

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
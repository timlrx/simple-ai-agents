## Simple AI Agents

Create simple multi-agent workflows using any LLMs - easy to experiment, use and deploy.

The package extends [Simple AI Chat][simple-ai-chat] by adding support for [100+ LLM providers][litellm], [structured responses][instructor] and multiple agents, similar to [Autogen][autogen]. With out of the box handling of LLM requests, session handling and structured response generation, multi-agent conversations can be easily orchestrated with Python to manage the control flow. This results in code which is easy to understand and extend with minimal dependencies.

__Note__: The package is in active development and the API is subject to change. Feedback and contributions are welcome!

## Features

- Mix and match LLM providers ([OpenAI, Huggingface, Ollama, Anthropic and more!][litellm]).
- Create and run chats with only a few lines of code!
- Integrates with [instructor][instructor] to provide structured responses for almost all models.
- Supports tool usage in models from Open AI, Azure, Anthropic, Bedrock, Vertex AI, Grok and Cerebras as well as selected Github, Together AI and Ollama models.
- Run multiple independent chats at once or create [Autogen][autogen] like multi-agent conversations.
- Minimal codebase: no code dives to figure out what's going on under the hood needed!
- Async and streaming for text response and structured response generation.
- Interactive CLI.

## Getting Started

Install the package using pip:

```bash
pip install simple-ai-agents
```

Set up the necessary environment variables for the LLM providers you want to use. Add the necessary environment variables corresponding to the [LLM providers](https://docs.litellm.ai/docs/providers) that will be used. Refer to `.env.example` for an example of the variables to add:

```
OPENAI_API_KEY=<your openai api key>
OPENAI_ORGANIZATION=<your openai organization>
HUGGINGFACE_API_KEY=<your huggingfacehub api token>
```

Use the client library easily call various LLM providers. The simplest way to get started is to create a `ChatLLMSession`. You can configure various options with with the `LLMOptions` typed dictionary:

```py
from simple_ai_agents.chat_session import ChatLLMSession
from simple_ai_agents.models import LLMOptions

openai = LLMOptions(model="gpt-4o-mini", temperature=0.7)
sess = ChatLLMSession(llm_options=openai)
prompt = "Why is the sky blue?"
response = sess.gen(prompt)
```

Overview of the main methods:

- `gen`: Generate a response synchronously. Supports passing in `tools` for tool usage.
- `gen_async`: Asynchronous version of `gen`.
- `gen_model`: Generate a structured response. Selects the best option for the provider and model.
- `gen_model_async`: Asynchronous version of `gen_model`.
- `stream`: Stream the response. Supports passing in `tools` for tool usage.
- `stream_async`: Asynchronous version of `stream`.
- `stream_model`: Stream the structured response. Selects the best option for the provider and model.
- `stream_model_async`: Asynchronous version of `stream_model`.


## Creating Agents

 `ChatAgent` extends on `ChatLLMSession`, by adding simple session handling capabilities, console printing and some nice syntax glue to allow calling the object directly to proxy the `gen` method. This allows for easy creation of chatbots and multi-agent conversations.

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
- `--model`: Specify the LLM model e.g. gpt-4o-mini, ollama/mistral etc. Uses gpt-4o-mini by default.
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

### Structured responses

To generate a structured response, use the `gen_model` method:

```py
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")

chatbot = ChatAgent(llm_options=openai)
parsed = chatbot.gen_model(
    "Extract `My name is John and I am 18 years old` into JSON",
    response_model=Person
)
```

The package automatically selects the best mode to generate JSON for a given provider and model. For the highest quality and reliability of structured responses, choose a model that supports tool usage.

[Tool usage][openai tools] is currently supported in Open AI, Azure, Anthropic, Bedrock, Vertex AI and Grok models. Selected [Ollama models][ollama tools] and [Together AI models][together json] also support structured response generation. For other providers and models, structured response is obtained by parsing the returned message results. This might result in a lower quality and accuracy of the structured response.

### Examples

- [Basic chatbot session](examples/sessions.py)
- [Multiple chatbot sessions](examples/chatbot_session.py)
- [Multi-agent conversation with different models](examples/multiple_agents.py)
- [Structured responses](examples/structured_responses.py)
- [LLM as Judge](examples/llm_judge.py)
- [Streaming with FastAPI](examples/fastapi_stream.py)

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
[openai tools]: https://platform.openai.com/docs/assistants/tools
[anyscale function calling]: https://docs.endpoints.anyscale.com/guides/function-calling/
[ollama json]: https://github.com/jmorganca/ollama/blob/main/docs/api.md#json-mode
[ollama tools]: https://ollama.com/search?c=tools
[together json]: https://docs.together.ai/docs/json-mode#supported-models
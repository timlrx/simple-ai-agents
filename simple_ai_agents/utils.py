import json
import logging
from typing import AsyncIterator, Iterator, Literal, Optional, TypeVar

from instructor import handle_response_model
from instructor.mode import Mode
from instructor.process_response import process_response, process_response_async
from pydantic import BaseModel

from simple_ai_agents.external import create_schema_from_function
from simple_ai_agents.models import Tool

T_Model = TypeVar("T_Model", bound=BaseModel)

# Taken from https://ollama.com/search?c=tools
ollama_tool_models = [
    "ollama/llama3.1",
    "ollama/mistral-nemo",
    "ollama/mistral-large",
    "ollama/qwen2",
    "ollama/mistral",
    "ollama/mixtral",
    "ollama/command-r",
    "ollama/command-r-plus",
    "ollama/hermes3",
    "ollama/llama3-groq-tool-use",
]

# https://docs.together.ai/docs/json-mode#supported-models
together_ai_tool_models = [
    "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "together_ai/mistralai/Mistral-7B-Instruct-v0.1",
]


def getJSONMode(llm_provider: str, model: str, stream: Optional[bool]) -> Mode:
    # LiteLLM transforms openai tools to anthropic / vertex hence no need for separate mode
    if llm_provider in [
        "openai",
        "azure",
        "anthropic",
        "bedrock",
        "vertex_ai",
        "groq",
    ]:
        return Mode.TOOLS
    # For together ai, json schema mode is more similar rather than tools
    elif llm_provider == "together_ai" and model in together_ai_tool_models:
        return Mode.JSON_SCHEMA
    # Not able to get the command-r and llama models to work with tools
    elif llm_provider == "github" and "gpt" in model:
        return Mode.TOOLS
    elif (
        llm_provider == "ollama" or llm_provider == "ollama_chat"
    ) and model in ollama_tool_models:
        return Mode.TOOLS
    # Stream with json not supported
    # https://inference-docs.cerebras.ai/openai#currently-unsupported-openai-features
    elif llm_provider == "cerebras" and not stream:
        return Mode.JSON_SCHEMA
    elif llm_provider == "ollama" or llm_provider == "ollama_chat":
        return Mode.JSON
    elif llm_provider == "anyscale" and model.replace("anyscale/", "") in [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ]:
        return Mode.JSON_SCHEMA
    else:
        logging.warning(
            f"{model} does not have support for any JSON mode. Defaulting to MD_JSON."
        )
        return Mode.MD_JSON


def format_tool_call(message: BaseModel | dict) -> dict:
    """
    Format the tool call message to be sent to the LLM
    """
    if not isinstance(message, (BaseModel, dict)):
        raise ValueError("Message should be a BaseModel or a dict.")

    d = message.model_dump() if isinstance(message, BaseModel) else message
    tool_calls = d.get("tool_calls")
    role = d.get("role")
    content = d.get("content")

    if not tool_calls or not role:
        raise ValueError("Message should have both 'tool_calls' and 'role' fields.")

    return {"role": role, "tool_calls": tool_calls, "content": content}


def process_json_response(
    response,
    response_model: type[T_Model],
    llm_provider: Optional[str],
    stream: bool,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
) -> T_Model:
    """
    Wrapper around instructor process_response to specially handle different response mode.
    If ollama is used with JSON mode, we patch the name and parse it as if it is a function call.
    """
    if response_model is not None:
        if mode in {Mode.JSON} and (
            llm_provider == "ollama" or llm_provider == "ollama_chat"
        ):
            message = response.choices[0].message
            tool_call = message.tool_calls[0]
            tool_call.function.name = response_model.openai_schema["name"]  # type: ignore
            return process_response(
                response,
                response_model=response_model,
                stream=stream,
                strict=strict,
                mode=Mode.TOOLS,
            )  # type: ignore
        else:
            # Let instructor handle the response
            return process_response(
                response,
                response_model=response_model,
                stream=stream,
                strict=strict,
                mode=mode,
            )  # type: ignore


async def process_json_response_async(
    response,
    response_model: type[T_Model],
    llm_provider: Optional[str],
    stream: bool,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
) -> T_Model:
    """
    Wrapper around instructor process_response to specially handle different response mode.
    If ollama is used with JSON mode, we patch the name and parse it as if it is a function call.
    """
    if response_model is not None:
        if mode in {Mode.JSON} and (
            llm_provider == "ollama" or llm_provider == "ollama_chat"
        ):
            message = response.choices[0].message
            tool_call = message.tool_calls[0]
            tool_call.function.name = response_model.openai_schema["name"]  # type: ignore
            return await process_response_async(
                response,
                response_model=response_model,
                stream=stream,
                strict=strict,
                mode=Mode.TOOLS,
            )  # type: ignore
        else:
            # Let instructor handle the response
            return await process_response_async(
                response,
                response_model=response_model,
                stream=stream,
                strict=strict,
                mode=mode,
            )  # type: ignore


def format_tool_schema(tools: list[Tool]):
    """
    Convert pydantic model to openai format dict for tools
    """
    for tool in tools:
        if isinstance(tool.tool_model, type) and issubclass(tool.tool_model, BaseModel):
            _, kwargs = handle_response_model(
                response_model=tool.tool_model, mode=Mode.TOOLS
            )
            tool.tool_model = kwargs["tools"][0]
        elif tool.tool_model is None and callable(tool.function):
            model = create_schema_from_function(tool.function.__name__, tool.function)
            _, kwargs = handle_response_model(response_model=model, mode=Mode.TOOLS)
            tool.tool_model = kwargs["tools"][0]

    tool_schemas = [tool.tool_model for tool in tools]
    return tools, tool_schemas


async def async_pydantic_to_text_stream(
    stream: AsyncIterator[BaseModel], mode: Literal["full", "delta"] = "full"
) -> AsyncIterator[str]:
    """
    Asynchronously converts a stream of Pydantic models to a stream of JSON strings,
    outputting either the full matching prefix or only the incremental difference
    between consecutive streams with changes.

    Args:
        stream (AsyncIterator[BaseModel]): An async iterator yielding Pydantic model instances.
        mode (Literal['full', 'delta']): The output mode. 'full' returns the full matching prefix
            (default). 'delta' returns only the incremental difference.

    Yields:
        AsyncIterator[str]: JSON string representation of the diff between consecutive Pydantic
        models.
    """
    json_history = []
    last_output = ""

    def get_matching_prefix(s1: str, s2: str) -> str:
        """Return the matching prefix of two strings."""
        for i, (c1, c2) in enumerate(zip(s1, s2)):
            if c1 != c2:
                return s1[:i]
        return s1  # If one string is a prefix of the other, return the shorter one

    async for model in stream:
        # Convert the Pydantic model to a JSON string
        new_json = json.dumps(model.model_dump(exclude_unset=True))

        # If this is not the first item and the new JSON is not in the history
        if json_history and new_json not in json_history:
            # Get the matching prefix with the last JSON in history
            diff = get_matching_prefix(json_history[-1], new_json)

            # Determine the output based on the mode
            if mode == "full":
                output = diff
            else:  # mode == 'delta'
                output = diff[len(last_output) :]

            # Yield the output if it's not empty
            if output:
                yield output
                last_output = diff

        # Update the JSON history
        json_history.append(new_json)

    # Ensure we output the last stream if it's different from the last output
    if json_history and json_history[-1] != last_output:
        if mode == "full":
            yield json_history[-1]
        else:  # mode == 'delta'
            yield json_history[-1][len(last_output) :]


def pydantic_to_text_stream(
    stream: Iterator[BaseModel], mode: Literal["full", "delta"] = "delta"
) -> Iterator[str]:
    """
    Converts a stream of Pydantic models to a stream of JSON strings,
    outputting either the full matching prefix or only the incremental difference
    between consecutive streams with changes.

    Args:
        stream (Iterator[BaseModel]): An iterator yielding Pydantic model instances.
        mode (Literal['full', 'delta']): The output mode.
            'full' returns the full matching prefix (default).
            'delta' returns only the incremental difference.

    Yields:
        str: JSON string representation of the diff between consecutive Pydantic models.
    """
    json_history = []
    last_output = ""

    def get_matching_prefix(s1: str, s2: str) -> str:
        """Return the matching prefix of two strings."""
        for i, (c1, c2) in enumerate(zip(s1, s2)):
            if c1 != c2:
                return s1[:i]
        return s1  # If one string is a prefix of the other, return the shorter one

    for model in stream:
        # Convert the Pydantic model to a JSON string
        new_json = json.dumps(model.model_dump(exclude_unset=True))

        # If this is not the first item and the new JSON is not in the history
        if json_history and new_json not in json_history:
            # Get the matching prefix with the last JSON in history
            diff = get_matching_prefix(json_history[-1], new_json)

            # Determine the output based on the mode
            if mode == "full":
                output = diff
            else:
                output = diff[len(last_output) :]

            if output:
                yield output
                last_output = diff

        # Update the JSON history
        json_history.append(new_json)

    # Ensure we output the last stream if it's different from the last output
    if json_history and json_history[-1] != last_output:
        if mode == "full":
            yield json_history[-1]
        else:
            yield json_history[-1][len(last_output) :]

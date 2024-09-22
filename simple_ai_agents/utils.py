import logging
from typing import Optional, TypeVar

from instructor import handle_response_model
from instructor.mode import Mode
from instructor.process_response import process_response
from pydantic import BaseModel

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


def getJSONMode(llm_provider: Optional[str], model: str) -> Mode:
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
    tool_schemas = [tool.tool_model for tool in tools]
    return tools, tool_schemas

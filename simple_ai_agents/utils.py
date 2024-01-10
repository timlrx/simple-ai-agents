from typing import Optional, Type, TypeVar

from instructor import OpenAISchema
from instructor.function_calls import Mode
from instructor.patch import process_response

T = TypeVar("T", bound=OpenAISchema)


def getJSONMode(llm_provider: Optional[str]) -> Mode:
    if llm_provider == "openai" or llm_provider == "azure":
        return Mode.TOOLS
    elif llm_provider == "ollama" or llm_provider == "ollama_chat":
        return Mode.JSON
    else:
        raise Exception(f"{llm_provider} does not have support for any JSON mode.")


def process_json_response(
    response,
    response_model: Type[OpenAISchema],
    llm_provider: Optional[str],
    stream: bool,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
) -> Type[T]:
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
            tool_call.function.name = response_model.openai_schema["name"]  # Patch name
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

import json
from json import JSONDecodeError
from typing import Any, AsyncGenerator, Generator, Literal, Optional, TypeVar, Union

import litellm
from instructor import OpenAISchema, Partial
from instructor.mode import Mode
from instructor.process_response import handle_response_model
from litellm import CustomStreamWrapper, acompletion, completion
from litellm.files.main import ModelResponse
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from pydantic import BaseModel, ValidationError

from simple_ai_agents.models import ChatMessage, ChatSession, LLMOptions, Tool
from simple_ai_agents.utils import (
    format_tool_call,
    format_tool_schema,
    getJSONMode,
    process_json_response,
    process_json_response_async,
)

litellm.telemetry = False
litellm.add_function_to_prompt = False  # add function to prompt for non openai models
litellm.drop_params = True  # drop params if unsupported by provider
litellm.suppress_debug_info = True

T_Model = TypeVar("T_Model", bound=BaseModel)


class ChatLLMSession(ChatSession):
    system: str = "You are a helpful assistant."
    llm_options: Optional[LLMOptions] = {"model": "gpt-4o-mini"}

    def prepare_request(
        self,
        prompt: str,
        system: Optional[str] = None,
        response_model: Optional[type[BaseModel]] = None,
        llm_options: Optional[LLMOptions] = None,
        stream: Optional[bool] = False,
    ) -> tuple[
        str,
        dict[str, Any],
        list[dict[str, Any]],
        ChatMessage,
        Optional[str],
        Optional[type[OpenAISchema]],
        Optional[Union[Mode, Literal["auto"]]],
    ]:
        """
        Prepare a request to send to liteLLM.

        Args:
            prompt (str): User prompt
            system (str, optional): System prompt. Defaults to None.
            response_model (BaseModel), optional:
                The response model to use for parsing the response.
                Defaults to None.
            llm_options (LLMOptions, optional): LiteLLM options.
                See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for details.
                Defaults to None.

        Returns:
            Tuple: (model, kwargs, history, user_message, llm_provider, response_model, mode)
        """
        mode = "auto"
        # Just use user prompt, no system prompt required
        if response_model:
            history = [
                {"role": "user", "content": prompt},
            ]
        # If saved messages exist, append it to prompt
        elif self.messages:
            history = [{"role": "system", "content": system or self.system}]
            for msg in self.messages:
                history.append({"role": msg.role, "content": msg.content})
            history.append({"role": "user", "content": prompt})
        else:
            history = [
                {"role": "system", "content": system or self.system},
                {"role": "user", "content": prompt},
            ]
        user_message = ChatMessage(
            role="user",
            content=prompt,
        )

        if llm_options:
            litellm_options: LLMOptions = llm_options
        elif self.llm_options:
            litellm_options: LLMOptions = self.llm_options  # type: ignore
        else:
            raise ValueError("No LLM options provided.")

        model = litellm_options.get("model") or ""
        _, custom_llm_provider, _, _ = get_llm_provider(model)
        if not model:
            raise ValueError("No LLM model provided.")
        kwargs = {k: v for k, v in litellm_options.items() if k != "model"}

        if response_model:
            mode = getJSONMode(custom_llm_provider, model, stream)
            kwargs["messages"] = history  # handle_response_model will add messages
            response_model, fn_kwargs = handle_response_model(
                response_model=response_model, mode=mode, **kwargs
            )
            if mode in {Mode.JSON, Mode.JSON_SCHEMA, Mode.MD_JSON}:
                history = fn_kwargs["messages"]
            # Add functions and function_call to kwargs
            kwargs.update(fn_kwargs)
            del kwargs["messages"]
        return (
            model,
            kwargs,
            history,
            user_message,
            custom_llm_provider,
            response_model,  # type: ignore
            mode,
        )

    def gen(
        self,
        prompt: str,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        tools: Optional[list[Tool]] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        llm_options: Optional[LLMOptions] = None,
    ) -> str:
        """
        Generate a chat response from the LLM.

        Args:
            prompt (str): User prompt
            system (str, optional): System prompt. Defaults to None.
            save_messages (bool, optional): Whether to save the messages.
                Defaults to None.
            tools (list[Tool], optional): List of tools available for use by the LLM.
                Defaults to None.
            tool_choice (str | dict[str, Any] | None, optional): Tool choice configuration.
                Defaults to individual providers default behaviour.
            llm_options (LLMOptions, optional): LiteLLM options.
                See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for details.
                Defaults to None.
        """
        model, kwargs, history, user_message, llm_provider, _, _ = self.prepare_request(
            prompt,
            system=system,
            llm_options=llm_options,
        )
        tools, tool_schemas = format_tool_schema(tools) if tools else (None, None)
        response: ModelResponse = completion(
            model=model,
            messages=history,
            tools=tool_schemas,
            tool_choice=tool_choice,
            **kwargs,
        )  # type: ignore

        # Make a 2nd request if tool calls are present
        response_message = response.choices[0]["message"]
        tool_calls = response_message["tool_calls"]
        if tool_calls and tools:
            tool_history = self._handle_tool_response(
                response_message, tool_calls, tools
            )
            history.extend(tool_history)
            response: ModelResponse = completion(
                model=model,
                messages=history,
                tools=tool_schemas if llm_provider == "anthropic" else None,
            )  # type: ignore
        try:
            content, assistant_message = self._handle_message_response(response)
            self.add_messages([user_message, assistant_message], save_messages)
        except KeyError:
            raise KeyError(f"No AI generation: {response}")
        return content

    async def gen_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        tools: Optional[list[Tool]] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        llm_options: Optional[LLMOptions] = None,
    ) -> str:
        """
        Generate a chat response from the LLM.

        Args:
            prompt (str): User prompt
            system (str, optional): System prompt. Defaults to None.
            save_messages (bool, optional): Whether to save the messages.
                Defaults to None.
            tools (list[Tool], optional): List of tools available for use by the LLM.
                Defaults to None.
            tool_choice (str | dict[str, Any] | None, optional): Tool choice configuration.
                Defaults to individual providers default behaviour.
            llm_options (LLMOptions, optional): LiteLLM options.
                See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for details.
                Defaults to None.
        """
        model, kwargs, history, user_message, llm_provider, _, _ = self.prepare_request(
            prompt, system=system, llm_options=llm_options
        )
        tools, tool_schemas = format_tool_schema(tools) if tools else (None, None)
        response: ModelResponse = await acompletion(
            model=model,
            messages=history,
            tools=tool_schemas,
            tool_choice=tool_choice,  # type: ignore
            **kwargs,
        )  # type: ignore
        response_message = response.choices[0]["message"]
        tool_calls = response_message["tool_calls"]
        if tool_calls and tools:
            tool_history = self._handle_tool_response(
                response_message, tool_calls, tools
            )
            history.extend(tool_history)
            response: ModelResponse = await acompletion(
                model=model,
                messages=history,
                tools=tool_schemas if llm_provider == "anthropic" else None,
            )  # type: ignore
        try:
            content, assistant_message = self._handle_message_response(response)
            self.add_messages([user_message, assistant_message], save_messages)
        except KeyError:
            raise KeyError(f"No AI generation: {response}")

        return content

    async def gen_model_async(
        self,
        prompt: str,
        response_model: type[T_Model | OpenAISchema | BaseModel],
        system: Optional[str] = None,
        llm_options: Optional[LLMOptions] = None,
        validation_retries: int = 1,
        strict: Optional[bool] = None,
    ) -> T_Model:  # type: ignore
        """
        Generate a response from the AI and parse it into a response model.

        Args:
            prompt (str): User prompt
            system (str, optional): System prompt. Defaults to None.
            llm_options (LLMOptions, optional): LiteLLM options.
                See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for details.
                Defaults to None.
            validation_retries (int, optional):
                Number of times to retry generating a valid response model.
            response_model (BaseModel): The response model to use for parsing the response
            strict (bool, optional): Whether to use strict json parsing. Defaults to None.

        Returns:
            Type[T]: An instance of the response model
        """
        if not response_model:
            raise ValueError("No response model provided.")
        (
            model,
            kwargs,
            history,
            user_message,
            llm_provider,
            response_model_processed,
            response_mode,
        ) = self.prepare_request(
            prompt,
            system=system,
            response_model=response_model,
            llm_options=llm_options,
        )  # type: ignore
        response_model_processed: T_Model
        response_mode: Mode
        retries = 0
        while retries <= validation_retries:
            # Excepts ValidationError, and JSONDecodeError
            try:
                response: ModelResponse = await acompletion(
                    model=model, messages=history, **kwargs  # type: ignore
                )
                model: T_Model = await process_json_response_async(
                    response,
                    response_model=response_model_processed,  # type: ignore
                    llm_provider=llm_provider,
                    stream=False,
                    strict=strict,
                    mode=response_mode,
                )
                self.total_prompt_length += response["usage"]["prompt_tokens"]
                self.total_completion_length += response["usage"]["completion_tokens"]
                self.total_length += response["usage"]["total_tokens"]
                return model
            except (ValidationError, JSONDecodeError) as e:
                if (
                    response_mode == Mode.TOOLS
                    or llm_provider == "ollama"
                    or llm_provider == "ollama_chat"
                ):
                    tool_call = response.choices[0].message.tool_calls[0]  # type: ignore
                    incorrect_json = tool_call.function.arguments
                    history.append({"role": "assistant", "content": incorrect_json})
                    history.append(
                        {
                            "role": "user",
                            "content": f"Recall the function correctly, exceptions found\n{e}",
                        }
                    )
                # For md_json, fresh retry work better - incorrect json is typically quite mangled
                retries += 1
                if retries > validation_retries:
                    raise e

    def gen_model(
        self,
        prompt: str,
        response_model: type[T_Model | OpenAISchema | BaseModel],
        system: Optional[str] = None,
        llm_options: Optional[LLMOptions] = None,
        validation_retries: int = 1,
        strict: Optional[bool] = None,
    ) -> T_Model:  # type: ignore
        """
        Generate a response from the AI and parse it into a response model.

        Args:
            prompt (str): User prompt
            system (str, optional): System prompt. Defaults to None.
            llm_options (LLMOptions, optional): LiteLLM options.
                See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for details.
                Defaults to None.
            validation_retries (int, optional):
                Number of times to retry generating a valid response model.
            response_model (BaseModel): The response model to use for parsing the response
            strict (bool, optional): Whether to use strict json parsing. Defaults to None.

        Returns:
            Type[T]: An instance of the response model
        """
        if not response_model:
            raise ValueError("No response model provided.")
        (
            model,
            kwargs,
            history,
            user_message,
            llm_provider,
            response_model_processed,
            response_mode,
        ) = self.prepare_request(
            prompt,
            system=system,
            response_model=response_model,
            llm_options=llm_options,
        )  # type: ignore
        response_model_processed: T_Model
        response_mode: Mode
        retries = 0
        while retries <= validation_retries:
            # Excepts ValidationError, and JSONDecodeError
            try:
                response: ModelResponse = completion(
                    model=model, messages=history, **kwargs  # type: ignore
                )
                model: T_Model = process_json_response(
                    response,
                    response_model=response_model_processed,  # type: ignore
                    llm_provider=llm_provider,
                    stream=False,
                    strict=strict,
                    mode=response_mode,
                )
                self.total_prompt_length += response["usage"]["prompt_tokens"]
                self.total_completion_length += response["usage"]["completion_tokens"]
                self.total_length += response["usage"]["total_tokens"]
                return model
            except (ValidationError, JSONDecodeError) as e:
                if (
                    response_mode == Mode.TOOLS
                    or llm_provider == "ollama"
                    or llm_provider == "ollama_chat"
                ):
                    tool_call = response.choices[0].message.tool_calls[0]  # type: ignore
                    incorrect_json = tool_call.function.arguments
                    history.append({"role": "assistant", "content": incorrect_json})
                    history.append(
                        {
                            "role": "user",
                            "content": f"Recall the function correctly, exceptions found\n{e}",
                        }
                    )
                # For md_json, fresh retry work better - incorrect json is typically quite mangled
                retries += 1
                if retries > validation_retries:
                    raise e

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        tools: Optional[list[Tool]] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        llm_options: Optional[LLMOptions] = None,
    ) -> Generator[dict[str, str], None, None]:
        """
        Generate a streaming response from the LLM.
        Stream response contains "delta" and "response" keys.
        - `delta` - latest response from the LLM model.
        - `response` - contains the entire conversation history up to that point.

        Args:
            prompt (str): User prompt
            system (str, optional): System prompt. Defaults to None.
            save_messages (bool, optional): Whether to save the messages.
                Defaults to None.
            tools (list[Tool], optional): List of tools available for use by the LLM.
                Defaults to None.
            tool_choice (str | dict[str, Any] | None, optional): Tool choice configuration.
                Defaults to individual providers default behaviour.
            llm_options (LLMOptions, optional): LiteLLM options.
                See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for details.
                Defaults to None.

        Yields:
            Generator[dict[str, str], None, None]: A generator yielding delta and response.
        """
        model, kwargs, history, user_message, llm_provider, _, _ = self.prepare_request(
            prompt, system=system, llm_options=llm_options
        )
        tools, tool_schemas = format_tool_schema(tools) if tools else (None, None)
        response = completion(
            model=model,
            messages=history,
            stream=True,
            tools=tool_schemas,
            tool_choice=tool_choice,
            **kwargs,
        )
        tool_calls = []
        content_chunks = []
        for chunk in response:
            delta = chunk["choices"][0]["delta"]  # type: ignore
            if delta and delta.get("content"):  # Text stream returned
                content = delta.get("content")
                content_chunks.append(content)
                yield {"delta": content, "response": "".join(content_chunks)}

            # Handle tool call
            if delta and delta.get("tool_calls") and tools:
                tc_chunk_list = delta.get("tool_calls")
                for tc_chunk in tc_chunk_list:
                    if len(tool_calls) <= tc_chunk.index:
                        tool_calls.append(
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        )
                    tc = tool_calls[tc_chunk.index]
                    if tc_chunk.id:
                        tc["id"] += tc_chunk.id
                    if tc_chunk.function.name:
                        tc["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tc["function"]["arguments"] += tc_chunk.function.arguments

        response_message = {"tool_calls": tool_calls, "role": "assistant"}
        if tool_calls and tools:
            tool_history = self._handle_tool_response(
                response_message, tool_calls, tools
            )
            history.extend(tool_history)
            response = completion(
                model=model,
                messages=history,
                stream=True,
                tools=tool_schemas if llm_provider == "anthropic" else None,
                **kwargs,
            )
        for chunk in response:
            delta = chunk["choices"][0]["delta"]  # type: ignore
            if delta and delta.get("content"):
                content = delta.get("content")
                content_chunks.append(content)
                yield {"delta": content, "response": "".join(content_chunks)}

        content = "".join(content_chunks)
        # streaming does not currently return token counts
        assistant_message = ChatMessage(
            role="assistant",
            content=content,
        )
        self.add_messages([user_message, assistant_message], save_messages)

    def stream_model(
        self,
        prompt: str,
        response_model: type[T_Model | OpenAISchema | BaseModel],
        system: Optional[str] = None,
        llm_options: Optional[LLMOptions] = None,
        validation_retries: int = 1,
        strict: Optional[bool] = None,
    ) -> Generator[T_Model, None, None]:
        """
        Generate a response from the AI and parse it into a response model.

        Args:
            prompt (str): User prompt
            system (str, optional): System prompt. Defaults to None.
            llm_options (LLMOptions, optional): LiteLLM options.
                See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for details.
                Defaults to None.
            validation_retries (int, optional):
                Number of times to retry generating a valid response model.
            response_model (BaseModel): The response model to use for parsing the response
            strict (bool, optional): Whether to use strict json parsing. Defaults to None.

        Returns:
            Generator[T_Model, None, None]: A generator yielding instances of the response model.
        """
        if not response_model:
            raise ValueError("No response model provided.")
        (
            model,
            kwargs,
            history,
            user_message,
            llm_provider,
            response_model_processed,
            response_mode,
        ) = self.prepare_request(
            prompt,
            system=system,
            response_model=response_model,
            llm_options=llm_options,
            stream=True,
        )  # type: ignore
        response_model_processed: T_Model = Partial[response_model_processed]  # type: ignore
        response_mode: Mode
        retries = 0
        while retries <= validation_retries:
            try:
                response: ModelResponse = completion(
                    model=model, messages=history, stream=True, **kwargs  # type: ignore
                )
                model_stream = process_json_response(
                    response,
                    response_model=response_model_processed,  # type: ignore
                    llm_provider=llm_provider,
                    stream=True,
                    strict=strict,
                    mode=response_mode,
                )
                for chunk in model_stream:
                    yield chunk
                return
            except (ValidationError, JSONDecodeError) as e:
                print(f"Error: {e}")
                # Keep it simple for streaming retry - just retry the prompt
                retries += 1
                if retries > validation_retries:
                    raise e

    async def stream_model_async(
        self,
        prompt: str,
        response_model: type[T_Model | OpenAISchema | BaseModel],
        system: Optional[str] = None,
        llm_options: Optional[LLMOptions] = None,
        validation_retries: int = 1,
        strict: Optional[bool] = None,
    ) -> AsyncGenerator[T_Model, None]:
        """
        Generate a response from the AI and parse it into a response model.

        Args:
            prompt (str): User prompt
            system (str, optional): System prompt. Defaults to None.
            llm_options (LLMOptions, optional): LiteLLM options.
                See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for details.
                Defaults to None.
            validation_retries (int, optional):
                Number of times to retry generating a valid response model.
            response_model (BaseModel): The response model to use for parsing the response
            strict (bool, optional): Whether to use strict json parsing. Defaults to None.

        Returns:
            Generator[T_Model, None]: A async generator yielding instances of the response model.
        """
        if not response_model:
            raise ValueError("No response model provided.")
        (
            model,
            kwargs,
            history,
            user_message,
            llm_provider,
            response_model_processed,
            response_mode,
        ) = self.prepare_request(
            prompt,
            system=system,
            response_model=response_model,
            llm_options=llm_options,
            stream=True,
        )  # type: ignore
        response_model_processed: T_Model = Partial[response_model_processed]  # type: ignore
        response_mode: Mode
        retries = 0
        while retries <= validation_retries:
            try:
                response: CustomStreamWrapper = await acompletion(
                    model=model, messages=history, stream=True, **kwargs
                )  # type: ignore
                model = await process_json_response_async(
                    response,
                    response_model=response_model_processed,  # type: ignore
                    llm_provider=llm_provider,
                    stream=True,
                    strict=strict,
                    mode=response_mode,
                )
                async for chunk in model:
                    yield chunk
                return
            except (ValidationError, JSONDecodeError) as e:
                print(f"Error: {e}")
                # Keep it simple for streaming retry - just retry the prompt
                retries += 1
                if retries > validation_retries:
                    raise e

    async def stream_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        tools: Optional[list[Tool]] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        llm_options: Optional[LLMOptions] = None,
    ) -> AsyncGenerator[dict[str, str], None]:
        """
        Generate a streaming response from the LLM.
        Stream response contains "delta" and "response" keys.
        - `delta` - latest response from the LLM model.
        - `response` - contains the entire conversation history up to that point.

        Args:
            prompt (str): User prompt
            system (str, optional): System prompt. Defaults to None.
            tools (list[Tool], optional): List of tools available for use by the LLM.
                Defaults to None.
            tool_choice (str | dict[str, Any] | None, optional): Tool choice configuration.
                Defaults to individual providers default behaviour.
            save_messages (bool, optional): Whether to save the messages.
                Defaults to None.
            llm_options (LLMOptions, optional): LiteLLM options.
                See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for details.
                Defaults to None.

        Yields:
            AsyncGenerator[dict[str, str], None]
        """
        model, kwargs, history, user_message, llm_provider, _, _ = self.prepare_request(
            prompt, system=system, llm_options=llm_options, stream=True
        )
        tools, tool_schemas = format_tool_schema(tools) if tools else (None, None)
        response: CustomStreamWrapper = await acompletion(
            model=model,
            messages=history,
            stream=True,
            tools=tool_schemas,
            tool_choice=tool_choice,  # type: ignore
            **kwargs,
        )
        tool_calls = []
        content_chunks = []
        async for chunk in response:
            delta = chunk["choices"][0]["delta"]
            if delta and delta.get("content"):  # Text stream returned
                content = delta.get("content")
                content_chunks.append(content)
                yield {"delta": content, "response": "".join(content_chunks)}

            # Handle tool call
            if delta and delta.get("tool_calls") and tools:
                tc_chunk_list = delta.get("tool_calls")
                for tc_chunk in tc_chunk_list:
                    if len(tool_calls) <= tc_chunk.index:
                        tool_calls.append(
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        )
                    tc = tool_calls[tc_chunk.index]
                    if tc_chunk.id:
                        tc["id"] += tc_chunk.id
                    if tc_chunk.function.name:
                        tc["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tc["function"]["arguments"] += tc_chunk.function.arguments

        response_message = {"tool_calls": tool_calls, "role": "assistant"}
        if tool_calls and tools:
            tool_history = self._handle_tool_response(
                response_message, tool_calls, tools
            )
            history.extend(tool_history)
            response = await acompletion(
                model=model,
                messages=history,
                stream=True,
                tools=tool_schemas if llm_provider == "anthropic" else None,
                **kwargs,
            )  # type: ignore
        async for chunk in response:
            delta = chunk["choices"][0]["delta"]
            if delta and delta.get("content"):
                content = delta.get("content")
                content_chunks.append(content)
                yield {"delta": content, "response": "".join(content_chunks)}

        content = "".join(content_chunks)
        # streaming does not currently return token counts
        assistant_message = ChatMessage(
            role="assistant",
            content=content,
        )
        self.add_messages([user_message, assistant_message], save_messages)

    def _handle_message_response(self, response: ModelResponse):
        """
        Handle the response message from the LLM and return the content and assistant message.
        """
        content: str = response.choices[0]["message"]["content"]
        assistant_message = ChatMessage(
            role="assistant",
            content=content,
            finish_reason=response.choices[0]["finish_reason"],
            prompt_length=response["usage"]["prompt_tokens"],
            completion_length=response["usage"]["completion_tokens"],
            total_length=response["usage"]["total_tokens"],
        )
        self.total_prompt_length += response["usage"]["prompt_tokens"]
        self.total_completion_length += response["usage"]["completion_tokens"]
        self.total_length += response["usage"]["total_tokens"]
        return content, assistant_message

    def _handle_tool_response(self, response_message, tool_calls, tools: list[Tool]):
        """
        Handle the tool response from the LLM and return the tool history.
        """
        # Use tool.tool_model["function"]["name"] instead of too.function.__name__
        # since returned response is based on schema and there might be a name mismatch
        available_functions = {
            tool.tool_model["function"]["name"]: tool.function for tool in tools  # type: ignore
        }
        tool_history = [format_tool_call(response_message)]
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            function_to_call = available_functions.get(function_name)
            if not callable(function_to_call):
                raise ValueError(
                    f"Function '{function_name}' does not exist or is not callable"
                )
            function_response = function_to_call(**function_args)
            function_message = {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
            tool_history.append(function_message)
        return tool_history

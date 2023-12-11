from json import JSONDecodeError
from typing import Any, AsyncGenerator, Generator, Optional, Type, TypeVar

import litellm
from instructor.function_calls import Mode
from instructor.patch import handle_response_model, process_response
from litellm import ModelResponse, acompletion, completion
from pydantic import BaseModel, ValidationError

from simple_ai_agents.models import ChatMessage, ChatSession, LLMOptions

litellm.telemetry = False
litellm.add_function_to_prompt = True  # add function to prompt for non openai models
litellm.drop_params = True  # drop params if unsupported by provider
litellm.suppress_debug_info = True

T = TypeVar("T", bound=BaseModel)


class ChatLLMSession(ChatSession):
    system: str = "You are a helpful assistant."
    llm_options: Optional[LLMOptions] = {"model": "gpt-3.5-turbo"}

    def prepare_request(
        self,
        prompt: str,
        system: Optional[str] = None,
        response_model: Optional[Type[BaseModel]] = None,
        llm_options: Optional[LLMOptions] = None,
    ) -> tuple[
        str,
        dict[str, Any],
        list[dict[str, Any]],
        ChatMessage,
        Optional[Type[BaseModel]],
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
            Tuple: (model, kwargs, history, user_message, response_model)
        """
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

        model = litellm_options.get("model")
        if not model:
            raise ValueError("No LLM model provided.")
        kwargs = {k: v for k, v in litellm_options.items() if k != "model"}

        if response_model:
            response_model, fn_kwargs = handle_response_model(
                response_model=response_model, kwargs=kwargs, mode=Mode.FUNCTIONS
            )
            # Add functions and function_call to kwargs
            kwargs.update(fn_kwargs)
        return model, kwargs, history, user_message, response_model

    def gen(
        self,
        prompt: str,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        llm_options: Optional[LLMOptions] = None,
    ) -> str:
        """
        Generate a chat response from the LLM.

        Args:
            prompt (str): User prompt
            system (str, optional): System prompt. Defaults to None.
            save_messages (bool, optional): Whether to save the messages.
                Defaults to None.
            llm_options (LLMOptions, optional): LiteLLM options.
                See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for details.
                Defaults to None.
        """
        model, kwargs, history, user_message, _ = self.prepare_request(
            prompt,
            system=system,
            llm_options=llm_options,
        )
        response = completion(model=model, messages=history, **kwargs)  # type: ignore
        try:
            content: str = response.choices[0]["message"]["content"]
            assistant_message = ChatMessage(
                role="assistant",
                content=content,
                finish_reason=response.choices[0]["finish_reason"],
                prompt_length=response["usage"]["prompt_tokens"],
                completion_length=response["usage"]["completion_tokens"],
                total_length=response["usage"]["total_tokens"],
            )
            self.add_messages(user_message, assistant_message, save_messages)
            self.total_prompt_length += response["usage"]["prompt_tokens"]
            self.total_completion_length += response["usage"]["completion_tokens"]
            self.total_length += response["usage"]["total_tokens"]
        except KeyError:
            raise KeyError(f"No AI generation: {response}")

        return content

    async def gen_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        llm_options: Optional[LLMOptions] = None,
    ) -> str:
        """
        Generate a chat response from the LLM.

        Args:
            prompt (str): User prompt
            system (str, optional): System prompt. Defaults to None.
            save_messages (bool, optional): Whether to save the messages.
                Defaults to None.
            llm_options (LLMOptions, optional): LiteLLM options.
                See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for details.
                Defaults to None.
        """
        model, kwargs, history, user_message, _ = self.prepare_request(
            prompt, system=system, llm_options=llm_options
        )
        response: ModelResponse = await acompletion(
            model=model, messages=history, **kwargs
        )  # type: ignore
        try:
            content: str = response.choices[0]["message"]["content"]
            assistant_message = ChatMessage(
                role="assistant",
                content=content,
                finish_reason=response.choices[0]["finish_reason"],
                prompt_length=response["usage"]["prompt_tokens"],
                completion_length=response["usage"]["completion_tokens"],
                total_length=response["usage"]["total_tokens"],
            )
            self.add_messages(user_message, assistant_message, save_messages)
            self.total_prompt_length += response["usage"]["prompt_tokens"]
            self.total_completion_length += response["usage"]["completion_tokens"]
            self.total_length += response["usage"]["total_tokens"]
        except KeyError:
            raise KeyError(f"No AI generation: {response}")

        return content

    async def gen_model_async(
        self,
        prompt: str,
        response_model: Type[T],
        system: Optional[str] = None,
        llm_options: Optional[LLMOptions] = None,
        validation_retries: int = 1,
        strict: Optional[bool] = None,
    ) -> Type[T]:  # type: ignore
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
            response_model,
        ) = self.prepare_request(
            prompt,
            system=system,
            response_model=response_model,
            llm_options=llm_options,
        )  # type: ignore
        retries = 0
        while retries <= validation_retries:
            # Excepts ValidationError, and JSONDecodeError
            try:
                response: ModelResponse = await acompletion(
                    model=model, messages=history, **kwargs
                )  # type: ignore
                model: Type[T] = process_response(
                    response,
                    response_model=response_model,
                    stream=False,
                    strict=strict,
                    mode=Mode.FUNCTIONS,
                )  # type: ignore
                self.total_prompt_length += response["usage"]["prompt_tokens"]
                self.total_completion_length += response["usage"]["completion_tokens"]
                self.total_length += response["usage"]["total_tokens"]
            except (ValidationError, JSONDecodeError) as e:
                history.append(response.choices[0].message.model_dump())  # type: ignore
                history.append(
                    {
                        "role": "user",
                        "content": f"Recall the function correctly, exceptions found\n{e}",
                    }
                )
                retries += 1
                if retries > validation_retries:
                    raise e
            return model

    def gen_model(
        self,
        prompt: str,
        response_model: Type[T],
        system: Optional[str] = None,
        llm_options: Optional[LLMOptions] = None,
        validation_retries: int = 1,
        strict: Optional[bool] = None,
    ) -> Type[T]:  # type: ignore
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
            response_model,
        ) = self.prepare_request(
            prompt,
            system=system,
            response_model=response_model,
            llm_options=llm_options,
        )  # type: ignore
        retries = 0
        while retries <= validation_retries:
            # Excepts ValidationError, and JSONDecodeError
            try:
                response = completion(model=model, messages=history, **kwargs)  # type: ignore
                model: Type[T] = process_response(
                    response,
                    response_model=response_model,
                    stream=False,
                    strict=strict,
                    mode=Mode.FUNCTIONS,
                )  # type: ignore
                self.total_prompt_length += response["usage"]["prompt_tokens"]
                self.total_completion_length += response["usage"]["completion_tokens"]
                self.total_length += response["usage"]["total_tokens"]
            except (ValidationError, JSONDecodeError) as e:
                history.append(response.choices[0].message.model_dump())  # type: ignore
                history.append(
                    {
                        "role": "user",
                        "content": f"Recall the function correctly, exceptions found\n{e}",
                    }
                )
                retries += 1
                if retries > validation_retries:
                    raise e
            return model

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
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
            llm_options (LLMOptions, optional): LiteLLM options.
                See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for details.
                Defaults to None.

        Yields:
            Generator[dict[str, str], None, None]
        """
        model, kwargs, history, user_message, _ = self.prepare_request(
            prompt, system=system, llm_options=llm_options
        )

        response = completion(model=model, messages=history, stream=True, **kwargs)
        content_chunks = []
        for chunk in response:
            delta: str = chunk["choices"][0]["delta"].get("content")  # type: ignore
            if delta:
                content_chunks.append(delta)
                yield {"delta": delta, "response": "".join(content_chunks)}

        content = "".join(content_chunks)
        # streaming does not currently return token counts
        assistant_message = ChatMessage(
            role="assistant",
            content=content,
        )
        self.add_messages(user_message, assistant_message, save_messages)

    async def stream_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
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
            save_messages (bool, optional): Whether to save the messages.
                Defaults to None.
            llm_options (LLMOptions, optional): LiteLLM options.
                See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for details.
                Defaults to None.

        Yields:
            AsyncGenerator[dict[str, str], None]
        """
        model, kwargs, history, user_message, _ = self.prepare_request(
            prompt, system=system, llm_options=llm_options
        )

        response: ModelResponse = await acompletion(
            model=model, messages=history, stream=True, **kwargs
        )  # type: ignore
        content_chunks = []
        async for chunk in response:  # type: ignore
            delta: str = chunk["choices"][0]["delta"].get("content")
            if delta:
                content_chunks.append(delta)
                yield {"delta": delta, "response": "".join(content_chunks)}

        content = "".join(content_chunks)
        # streaming does not currently return token counts
        assistant_message = ChatMessage(
            role="assistant",
            content=content,
        )
        self.add_messages(user_message, assistant_message, save_messages)

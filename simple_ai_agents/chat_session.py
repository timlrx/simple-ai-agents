from typing import Any, Dict, Optional

import litellm
from litellm import ModelResponse, acompletion, completion

from simple_ai_agents.models import ChatMessage, ChatSession, LLMOptions

litellm.telemetry = False


class ChatLLMSession(ChatSession):
    system: str = "You are a helpful assistant."
    llm_options: Optional[LLMOptions] = {"model": "gpt-3.5-turbo"}

    def prepare_request(
        self,
        prompt: str,
        system: Optional[str] = None,
        llm_options: Optional[LLMOptions] = None,
    ):
        # If save_messages exist, append it to prompt
        if self.messages:
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
        other_options = {k: v for k, v in litellm_options.items() if k != "model"}
        return model, other_options, history, user_message

    def gen(
        self,
        prompt: str,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        llm_options: Optional[LLMOptions] = None,
    ):
        model, other_options, history, user_message = self.prepare_request(
            prompt, system=system, llm_options=llm_options
        )
        response = completion(model=model, messages=history, **other_options)  # type: ignore
        try:
            content = response.choices[0].message.content
            assistant_message = ChatMessage(
                role="assistant",
                content=content,
                finish_reason=response.choices[0].message.content,
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
        params: Optional[Dict[str, Any]] = None,
        llm_options: Optional[LLMOptions] = None,
    ):
        model, other_options, history, user_message = self.prepare_request(
            prompt, system=system, llm_options=llm_options
        )
        response: ModelResponse = await acompletion(
            model=model, messages=history, **other_options
        )  # type: ignore
        try:
            content = response.choices[0].message.content
            assistant_message = ChatMessage(
                role="assistant",
                content=content,
                finish_reason=response.choices[0].message.content,
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

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        llm_options: Optional[LLMOptions] = None,
    ):
        model, other_options, history, user_message = self.prepare_request(
            prompt, system=system, llm_options=llm_options
        )

        response = completion(
            model=model, messages=history, stream=True, **other_options  # type: ignore
        )
        content = []
        for chunk in response:
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                content.append(delta)
                yield {"delta": delta, "response": "".join(content)}

        # streaming does not currently return token counts
        assistant_message = ChatMessage(
            role="assistant",
            content="".join(content),
        )
        self.add_messages(user_message, assistant_message, save_messages)
        return

    async def stream_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        llm_options: Optional[LLMOptions] = None,
    ):
        model, other_options, history, user_message = self.prepare_request(
            prompt, system=system, llm_options=llm_options
        )

        response: ModelResponse = await acompletion(
            model=model, messages=history, stream=True, **other_options
        )  # type: ignore
        content = []
        async for chunk in response:  # type: ignore
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                content.append(delta)
                yield {"delta": delta, "response": "".join(content)}

        # streaming does not currently return token counts
        assistant_message = ChatMessage(
            role="assistant",
            content="".join(content),
        )
        self.add_messages(user_message, assistant_message, save_messages)
        return

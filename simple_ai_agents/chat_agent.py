from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel
from rich.console import Console

from simple_ai_agents.chat_session import ChatLLMSession
from simple_ai_agents.models import LLMOptions

# TODO: Add save, load


class ChatAgent(BaseModel):
    """
    A chatbot class that provides additional functionality
    for creating and managing chat sessions.

    Args:
        character (str, optional): The name of the chatbot for console display.
            Defaults to None.
        system (str, optional): System prompt for chatbot message.
            If None is provided it defaults to "You are a helpful assistant."
        id (Union[str, UUID], optional): Initial session ID of the chatbot.
            Defaults to uuid4().
        prime (bool, optional): Whether to prime the chatbot with initial messages.
            Defaults to True.
        default_session (bool, optional): Whether to create a default chat session.
            Defaults to True.
        console (bool, optional): Whether to enable interactive console mode.
            Defaults to False.
        **kwargs: Additional options to pass to the `new_session` method.
            To customize GPT options, pass a `llm_options` dictionary.

    Attributes:
        sessions (dict): A dictionary of chat sessions,
            where the keys are the session IDs and the values
            are the corresponding `ChatSession` objects.
        default_session (ChatSession): The default chat session.

    Methods:
        new_session: Creates a new chat session.
        interactive_console: Starts an interactive console for the chatbot.
    """

    default_session: Optional[ChatLLMSession]
    sessions: Dict[Union[str, UUID], ChatLLMSession] = {}

    def __init__(
        self,
        character: Optional[str] = None,
        system: Optional[str] = None,
        id: Union[str, UUID] = uuid4(),
        prime: bool = True,
        default_session: bool = True,
        console: bool = False,
        **kwargs,
    ):
        system_format = self.build_system(system)
        sessions = {}
        new_default_session = None
        super().__init__(default_session=new_default_session, sessions=sessions)

        if default_session:
            new_default_session = self.new_session(
                set_default=True, system=system_format, id=id, **kwargs
            )

        if console:
            if not new_default_session:
                raise ValueError(
                    "A default session needs to exists to run in interactive mode."
                )
            character = "Chat Agent" if not character else character
            new_default_session.title = character
            self.interactive_console(character=character, prime=prime)

    def new_session(
        self,
        set_default: bool = True,
        **kwargs,
    ) -> ChatLLMSession:
        sess = ChatLLMSession(**kwargs)
        self.sessions[sess.id] = sess
        if set_default:
            self.default_session = sess
        return sess

    def get_session(self, id: Optional[Union[str, UUID]] = None) -> ChatLLMSession:
        try:
            sess = self.sessions[id] if id else self.default_session
        except KeyError:
            raise KeyError("No session by that key exists.")
        if not sess:
            raise ValueError("No default session exists.")
        return sess

    def list_sessions(self) -> list[Union[str, UUID]]:
        return list(self.sessions.keys())

    def reset_session(self, id: Optional[Union[str, UUID]] = None) -> None:
        sess = self.get_session(id)
        sess.messages = []

    def delete_session(self, id: Optional[Union[str, UUID]] = None) -> None:
        sess = self.get_session(id)
        if self.default_session:
            if sess.id == self.default_session.id:
                self.default_session = None
        del self.sessions[sess.id]
        del sess

    @contextmanager
    def session(self, **kwargs):
        sess = self.new_session(set_default=True, **kwargs)
        try:
            yield sess
        finally:
            self.delete_session(sess.id)

    def __call__(
        self,
        prompt: Union[str, Any],
        id: Optional[Union[str, UUID]] = None,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        llm_options: Optional[LLMOptions] = None,
    ) -> str:
        """
        Generate a response from the AI.
        """
        sess = self.get_session(id)
        return sess.gen(
            prompt,
            system=system,
            save_messages=save_messages,
            params=params,
            llm_options=llm_options,
        )

    def stream(
        self,
        prompt: Union[str, Any],
        id: Optional[Union[str, UUID]] = None,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        llm_options: Optional[LLMOptions] = None,
    ):
        """
        Generate a response from the AI.
        """
        sess = self.get_session(id)
        return sess.stream(
            prompt,
            system=system,
            save_messages=save_messages,
            params=params,
            llm_options=llm_options,
        )

    def build_system(self, system: Optional[str] = None) -> str:
        default = "You are a helpful assistant."
        if system:
            return system
        else:
            return default

    def interactive_console(
        self, character: Optional[str] = None, prime: bool = True
    ) -> None:
        console = Console(highlight=False, force_jupyter=False)
        sess = self.default_session
        ai_text_color = "bright_magenta"

        if not sess:
            raise ValueError("No default session exists.")

        # prime with a unique starting response to the user
        if prime:
            console.print(f"[b]{character}[/b]: ", end="", style=ai_text_color)
            for chunk in sess.stream("Hello!"):
                console.print(chunk["delta"], end="", style=ai_text_color)

        while True:
            console.print()
            try:
                user_input = console.input("[b]You:[/b] ").strip()
                if not user_input:
                    break

                console.print(f"[b]{character}[/b]: ", end="", style=ai_text_color)
                for chunk in sess.stream(user_input):
                    console.print(chunk["delta"], end="", style=ai_text_color)
            except KeyboardInterrupt:
                break

    def __str__(self) -> str | None:
        if self.default_session:
            return self.default_session.json(exclude_none=True, indent=2)

    def __repr__(self) -> str:
        return ""

    # Tabulators for returning total token counts
    def message_totals(self, attr: str, id: Optional[Union[str, UUID]] = None) -> int:
        sess = self.get_session(id)
        return getattr(sess, attr)

    @property
    def total_prompt_length(self, id: Optional[Union[str, UUID]] = None) -> int:
        return self.message_totals("total_prompt_length", id)

    @property
    def total_completion_length(self, id: Optional[Union[str, UUID]] = None) -> int:
        return self.message_totals("total_completion_length", id)

    @property
    def total_length(self, id: Optional[Union[str, UUID]] = None) -> int:
        return self.message_totals("total_length", id)

    # alias total_tokens to total_length for common use
    @property
    def total_tokens(self, id: Optional[Union[str, UUID]] = None) -> int:
        return self.total_length(id)  # type: ignore


class ChatAgentAsync(ChatAgent):
    def __init__(
        self,
        character: Optional[str] = None,
        system: Optional[str] = None,
        id: Union[str, UUID] = uuid4(),
        prime: bool = True,
        default_session: bool = True,
        console: bool = False,
        **kwargs,
    ):
        super().__init__(
            character=character,
            system=system,
            id=id,
            prime=prime,
            default_session=default_session,
            console=console,
            **kwargs,
        )

    async def __call__(
        self,
        prompt: Union[str, Any],
        id: Optional[Union[str, UUID]] = None,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        llm_options: Optional[LLMOptions] = None,
    ) -> str:
        """
        Generate a response from the AI.
        """
        sess = self.get_session(id)
        return await sess.gen_async(
            prompt,
            system=system,
            save_messages=save_messages,
            params=params,
            llm_options=llm_options,
        )

    async def stream(
        self,
        prompt: Union[str, Any],
        id: Optional[Union[str, UUID]] = None,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        llm_options: Optional[LLMOptions] = None,
    ):
        """
        Generate a response from the AI.
        """
        sess = self.get_session(id)
        return sess.stream_async(
            prompt,
            system=system,
            save_messages=save_messages,
            params=params,
            llm_options=llm_options,
        )

    @asynccontextmanager
    async def session(self, **kwargs):
        sess = self.new_session(set_default=True, **kwargs)
        try:
            yield sess
        finally:
            self.delete_session(sess.id)

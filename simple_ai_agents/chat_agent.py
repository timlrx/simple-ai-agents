import csv
import datetime
import json
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, Optional, Type, TypeVar, Union
from uuid import UUID, uuid4

from dateutil import tz
from pydantic import BaseModel
from rich.console import Console

from simple_ai_agents.chat_session import ChatLLMSession
from simple_ai_agents.models import ChatMessage, LLMOptions

T = TypeVar("T", bound=BaseModel)


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
        character (str): The name of the chatbot for console display.
        ai_text_color (str): Print color of the chatbot's messages.

    Methods:
        new_session: Creates a new chat session.
        interactive_console: Starts an interactive console for the chatbot.
    """

    default_session: Optional[ChatLLMSession]
    sessions: Dict[Union[str, UUID], ChatLLMSession] = {}
    character: Optional[str] = "Chat Agent"
    ai_text_color: str = "bright_magenta"

    def __init__(
        self,
        character: Optional[str] = None,
        system: Optional[str] = None,
        id: Union[str, UUID] = uuid4(),
        prime: bool = True,
        default_session: bool = True,
        console: bool = False,
        ai_text_color: Optional[str] = None,
        display_names: bool = True,
        **kwargs,
    ):
        """
        Initialize a chatbot agent.

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
            ai_text_color (str, optional): Print color of the chatbot's messages.
            display_names (bool, optional):
                Whether to display character names in the console.
            **kwargs: Additional options to pass to the `new_session` method.
                To customize GPT options, pass a `llm_options` dictionary.
        """
        system_format = self.build_system(system)
        sessions = {}
        new_default_session = None
        super().__init__(default_session=new_default_session, sessions=sessions)

        if character:
            self.character = character
        if ai_text_color:
            self.ai_text_color = ai_text_color
        if default_session:
            new_default_session = self.new_session(
                set_default=True, system=system_format, id=id, **kwargs
            )
        if console:
            if not new_default_session:
                raise ValueError(
                    "A default session needs to exists to run in interactive mode."
                )
            new_default_session.title = character
            # print(kwargs)
            self.interactive_console(
                character=self.character,
                prime=prime,
                display_names=display_names,
                prompt=kwargs["prompt"] if "prompt" in kwargs else None,
            )

    def new_session(
        self,
        set_default: bool = True,
        **kwargs,
    ) -> ChatLLMSession:
        """
        Create a new chat session.

        Args:
            set_default (bool, optional): Whether to set the new session as the default.
                Defaults to True.
            **kwargs: Additional options to pass to the `ChatLLMSession` constructor.
        """
        sess = ChatLLMSession(**kwargs)
        self.sessions[sess.id] = sess
        if set_default:
            self.default_session = sess
        return sess

    def get_session(self, id: Optional[Union[str, UUID]] = None) -> ChatLLMSession:
        """
        Get a chat session by ID. If no ID is provided, return the default session.
        """
        try:
            sess = self.sessions[id] if id else self.default_session
        except KeyError:
            raise KeyError("No session by that key exists.")
        if not sess:
            raise ValueError("No default session exists.")
        return sess

    def list_sessions(self) -> list[Union[str, UUID]]:
        """
        List all session IDs.
        """
        return list(self.sessions.keys())

    def reset_session(self, id: Optional[Union[str, UUID]] = None) -> None:
        """
        Reset a chat session by ID.
        """
        sess = self.get_session(id)
        sess.messages = []

    def delete_session(self, id: Optional[Union[str, UUID]] = None) -> None:
        """
        Delete a chat session by ID.
        """
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
        llm_options: Optional[LLMOptions] = None,
        console_output: bool = False,
    ) -> str:
        """
        Generate a response from the AI.
        """
        sess = self.get_session(id)
        if console_output:
            console = Console(highlight=False, force_jupyter=False)
            ai_text_color = self.ai_text_color
            console.print(f"[b]{self.character}[/b]: ", end="", style=ai_text_color)
            stream = sess.stream(
                prompt,
                system=system,
                save_messages=save_messages,
                llm_options=llm_options,
            )
            for chunk in stream:
                console.print(chunk["delta"], end="", style=ai_text_color)
            console.print()
            return chunk["response"]  # type: ignore
        else:
            return sess.gen(
                prompt,
                system=system,
                save_messages=save_messages,
                llm_options=llm_options,
            )

    def stream(
        self,
        prompt: Union[str, Any],
        id: Optional[Union[str, UUID]] = None,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
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
            llm_options=llm_options,
        )

    def gen_model(
        self,
        prompt: Union[str, Any],
        response_model: Type[T],
        id: Optional[Union[str, UUID]] = None,
        system: Optional[str] = None,
        llm_options: Optional[LLMOptions] = None,
    ):
        """
        Generate a pydantic typed model from the AI.
        """
        sess = self.get_session(id)
        return sess.gen_model(
            prompt,
            response_model,
            system=system,
            llm_options=llm_options,
        )

    def build_system(self, system: Optional[str] = None) -> str:
        default = "You are a helpful assistant."
        if system:
            return system
        else:
            return default

    def interactive_console(
        self,
        character: Optional[str] = None,
        prime: bool = True,
        prompt: Optional[str] = None,
        display_names: bool = True,
    ) -> None:
        """
        Start an interactive console for the chatbot.
        """
        console = Console(highlight=False, force_jupyter=False)
        sess = self.default_session
        ai_text_color = self.ai_text_color
        user_prompt_suffix = "[b]You:[/b]" if display_names else "> "
        agent_prompt_suffix = (
            f"[b]{character}[/b]: " if display_names and character else ""
        )

        if not sess:
            raise ValueError("No default session exists.")

        # prime with a unique starting response to the user
        if prime:
            console.print(agent_prompt_suffix, end="", style=ai_text_color)
            for chunk in sess.stream("Hello!"):
                console.print(chunk["delta"], end="", style=ai_text_color)

        start = True
        while True:
            console.print()
            try:
                user_input = (
                    prompt
                    if start and prompt
                    else console.input(user_prompt_suffix).strip()
                )
                start = False
                if not user_input:
                    break

                console.print(agent_prompt_suffix, end="", style=ai_text_color)
                for chunk in sess.stream(user_input):
                    console.print(chunk["delta"], end="", style=ai_text_color)
            except KeyboardInterrupt:
                break

    def __str__(self) -> str | None:
        if self.default_session:
            return self.default_session.model_dump_json(exclude_none=True, indent=2)

    def __repr__(self) -> str:
        return ""

    def save_session(
        self,
        output_path: Optional[str] = None,
        id: Optional[Union[str, UUID]] = None,
        format: str = "csv",
    ):
        sess = self.get_session(id)
        sess_dict = sess.model_dump(
            exclude_none=True,
        )
        output_path = output_path or f"chat_session.{format}"
        if format == "csv":
            with open(output_path, "w", encoding="utf-8") as f:
                fields = [
                    "role",
                    "content",
                    "received_at",
                    "prompt_length",
                    "completion_length",
                    "total_length",
                    "finish_reason",
                ]
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for message in sess_dict["messages"]:
                    # datetime must be in common format to be loaded into spreadsheet
                    # for human-readability, the timezone is set to local machine
                    local_datetime = message["received_at"].astimezone()
                    message["received_at"] = local_datetime.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    w.writerow(message)
        elif format == "json":
            with open(output_path, "w") as f:
                f.write(sess.model_dump_json(exclude_none=True))

    def load_session(self, input_path: str, id: Union[str, UUID] = uuid4(), **kwargs):
        assert input_path.endswith(".csv") or input_path.endswith(
            ".json"
        ), "Only CSV and JSON imports are accepted."

        if input_path.endswith(".csv"):
            with open(input_path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                messages = []
                for row in r:
                    # need to convert the datetime back to UTC
                    local_datetime = datetime.datetime.strptime(
                        row["received_at"], "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=tz.tzlocal())
                    row["received_at"] = local_datetime.astimezone(
                        datetime.timezone.utc
                    )
                    # https://stackoverflow.com/a/68305271
                    row = {k: (None if v == "" else v) for k, v in row.items()}
                    messages.append(ChatMessage(**row))  # type: ignore

            sess = self.new_session(id=id, **kwargs)
            sess.messages = messages
            return sess

        if input_path.endswith(".json"):
            with open(input_path, "r") as f:
                sess_dict = json.loads(f.read())
            # update session with info not loaded, e.g. auth/api_url
            for arg in kwargs:
                sess_dict[arg] = kwargs[arg]
            sess = self.new_session(**sess_dict)
            return sess

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
        ai_text_color: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            character=character,
            system=system,
            id=id,
            prime=prime,
            default_session=default_session,
            console=console,
            ai_text_color=ai_text_color,
            **kwargs,
        )

    async def __call__(
        self,
        prompt: Union[str, Any],
        id: Optional[Union[str, UUID]] = None,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        llm_options: Optional[LLMOptions] = None,
    ) -> str:
        sess = self.get_session(id)
        return await sess.gen_async(
            prompt,
            system=system,
            save_messages=save_messages,
            llm_options=llm_options,
        )

    async def stream(
        self,
        prompt: Union[str, Any],
        id: Optional[Union[str, UUID]] = None,
        system: Optional[str] = None,
        save_messages: Optional[bool] = None,
        llm_options: Optional[LLMOptions] = None,
    ):
        sess = self.get_session(id)
        return sess.stream_async(
            prompt,
            system=system,
            save_messages=save_messages,
            llm_options=llm_options,
        )

    async def gen_model(
        self,
        prompt: Union[str, Any],
        response_model: Type[T],
        id: Optional[Union[str, UUID]] = None,
        system: Optional[str] = None,
        llm_options: Optional[LLMOptions] = None,
    ):
        sess = self.get_session(id)
        return await sess.gen_model_async(
            prompt,
            response_model,
            system=system,
            llm_options=llm_options,
        )

    @asynccontextmanager
    async def session(self, **kwargs):
        sess = self.new_session(set_default=True, **kwargs)
        try:
            yield sess
        finally:
            self.delete_session(sess.id)

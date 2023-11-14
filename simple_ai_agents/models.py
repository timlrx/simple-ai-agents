import datetime
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


def now_tz():
    return datetime.datetime.now(datetime.timezone.utc)


class LLMOptions(TypedDict, total=False):
    model: str
    functions: List
    function_call: str
    temperature: float
    top_p: float
    n: int
    stream: bool
    stop: str
    max_tokens: float
    presence_penalty: float
    frequency_penalty: float
    logit_bias: dict
    user: str
    deployment_id: str
    request_timeout: int
    api_base: str
    api_version: str
    api_key: str
    model_list: list


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[str] = None
    received_at: datetime.datetime = Field(default_factory=now_tz)
    finish_reason: Optional[str] = None
    prompt_length: Optional[int] = None
    completion_length: Optional[int] = None
    total_length: Optional[int] = None

    def __str__(self) -> str:
        return str(self.model_dump_json(exclude_none=True))


class ChatSession(BaseModel):
    id: Union[str, UUID] = Field(default_factory=uuid4)
    created_at: datetime.datetime = Field(default_factory=now_tz)
    system: str
    params: Dict[str, Any] = {}
    messages: List[ChatMessage] = []
    input_fields: Set[str] = set()
    recent_messages: Optional[int] = None
    save_messages: Optional[bool] = True
    total_prompt_length: int = 0
    total_completion_length: int = 0
    total_length: int = 0
    title: Optional[str] = None

    def __str__(self) -> str:
        sess_start_str = self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        if self.messages:
            last_message_str = self.messages[-1].received_at.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            last_message_str = "N/A"
        return f"""Chat session started at {sess_start_str}:
        - {len(self.messages):,} Messages
        - Last message sent at {last_message_str}"""

    def add_messages(
        self,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        save_messages: Optional[bool] = None,
    ) -> None:
        # if save_messages is explicitly defined, always use that choice
        # instead of the default
        to_save = isinstance(save_messages, bool)

        if to_save:
            if save_messages:
                self.messages.append(user_message)
                self.messages.append(assistant_message)
        elif self.save_messages:
            self.messages.append(user_message)
            self.messages.append(assistant_message)

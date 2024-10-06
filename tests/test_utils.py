from copy import deepcopy
from typing import Any, Optional, Tuple, Type

import pytest
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from simple_ai_agents.utils import (
    async_pydantic_to_text_stream,
    pydantic_to_text_stream,
)


def partial(model: Type[BaseModel]):
    def make_field_optional(
        field: FieldInfo, default: Any = None
    ) -> Tuple[Any, FieldInfo]:
        new = deepcopy(field)
        new.default = default
        new.annotation = Optional[field.annotation]  # type: ignore
        return new.annotation, new

    return create_model(
        f"Partial{model.__name__}",
        __base__=model,
        __module__=model.__module__,
        **{
            field_name: make_field_optional(field_info)
            for field_name, field_info in model.model_fields.items()
        },  # type: ignore
    )  # type: ignore


class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")


class PersonModel(BaseModel):
    person: Person


class MultiPersonModel(BaseModel):
    persons: list[Person]


PartialMultiPersonModel = partial(MultiPersonModel)
PartialPerson = partial(Person)
PartialPersonModel = partial(PersonModel)


def test_pydantic_to_text_stream():
    def sample_person_stream():
        PartialPersonModel = partial(PersonModel)
        PartialPerson = partial(Person)
        yield PartialPersonModel(person=PartialPerson(name=None, age=None))
        yield PartialPersonModel(person=PartialPerson(name="Alice", age=None))
        yield PartialPersonModel(person=PartialPerson(name="Alice", age=None))
        yield PartialPersonModel(person=PartialPerson(name="Alice", age=None))

    for i, diff in enumerate(
        pydantic_to_text_stream(sample_person_stream(), mode="full")
    ):
        if i == 0:
            assert diff == '{"person": {"name": '
        elif i == 1:
            assert diff == '{"person": {"name": "Alice", "age": null}}'

    for i, diff in enumerate(
        pydantic_to_text_stream(sample_person_stream(), mode="delta")
    ):
        if i == 0:
            assert diff == '{"person": {"name": '
        elif i == 1:
            assert diff == '"Alice", "age": null}}'


@pytest.mark.asyncio
async def test_async_pydantic_to_text_stream():
    async def sample_person_stream():
        PartialPersonModel = partial(PersonModel)
        PartialPerson = partial(Person)
        yield PartialPersonModel(person=PartialPerson(name=None, age=None))
        yield PartialPersonModel(person=PartialPerson(name="Alice", age=None))
        yield PartialPersonModel(person=PartialPerson(name="Alice", age=None))
        yield PartialPersonModel(person=PartialPerson(name="Alice", age=None))

    i = 0
    async for diff in async_pydantic_to_text_stream(
        sample_person_stream(), mode="full"
    ):
        if i == 0:
            assert diff == '{"person": {"name": '
        elif i == 1:
            assert diff == '{"person": {"name": "Alice", "age": null}}'
        i += 1

    i = 0
    async for diff in async_pydantic_to_text_stream(
        sample_person_stream(), mode="delta"
    ):
        if i == 0:
            assert diff == '{"person": {"name": '
        elif i == 1:
            assert diff == '"Alice", "age": null}}'
        i += 1


def test_pydantic_to_text_stream_list():
    def sample_multi_person_stream():
        yield PartialMultiPersonModel(persons=[PartialPerson(name="Alice", age=None)])
        yield PartialMultiPersonModel(persons=[PartialPerson(name="Alice", age=None)])
        yield PartialMultiPersonModel(
            persons=[
                PartialPerson(name="Alice", age=None),
                PartialPerson(name=None, age=None),
            ]
        )
        yield PartialMultiPersonModel(
            persons=[
                PartialPerson(name="Alice", age=None),
                PartialPerson(name=None, age=None),
            ]
        )
        yield PartialMultiPersonModel(
            persons=[
                PartialPerson(name="Alice", age=None),
                PartialPerson(name="Bob", age=None),
            ]
        )
        yield PartialMultiPersonModel(
            persons=[
                PartialPerson(name="Alice", age=None),
                PartialPerson(name="Bob", age=30),
            ]
        )

    for i, diff in enumerate(
        pydantic_to_text_stream(sample_multi_person_stream(), mode="full")
    ):
        if i == 0:
            assert diff == '{"persons": [{"name": "Alice", "age": null}'
        elif i == 1:
            assert diff == '{"persons": [{"name": "Alice", "age": null}, {"name": '
        elif i == 2:
            assert (
                diff
                == '{"persons": [{"name": "Alice", "age": null}, {"name": "Bob", "age": '
            )
        elif i == 3:
            assert (
                diff
                == '{"persons": [{"name": "Alice", "age": null}, {"name": "Bob", "age": 30}]}'
            )

    for i, diff in enumerate(
        pydantic_to_text_stream(sample_multi_person_stream(), mode="delta")
    ):
        if i == 0:
            assert diff == '{"persons": [{"name": "Alice", "age": null}'
        elif i == 1:
            assert diff == ', {"name": '
        elif i == 2:
            assert diff == '"Bob", "age": '
        elif i == 3:
            assert diff == "30}]}"

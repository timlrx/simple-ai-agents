import inspect
import textwrap
from typing import Any, Callable, Optional, Sequence, Union, overload

import pydantic
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from typing_extensions import Annotated, get_args, get_origin


def create_schema_from_function(
    model_name: str,
    func: Callable,
    *,
    filter_args: Optional[Sequence[str]] = None,
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = False,
    include_injected: bool = True,
) -> type[BaseModel]:
    """Create a pydantic schema from a function's signature.
    Adapted from: https://python.langchain.com/api_reference/_modules/langchain_core/tools/base.html

    Args:
        model_name: Name to assign to the generated pydantic schema.
        func: Function to generate the schema from.
        filter_args: Optional list of arguments to exclude from the schema.
            Defaults to FILTERED_ARGS.
        parse_docstring: Whether to parse the function's docstring for descriptions
            for each argument. Defaults to False.
        error_on_invalid_docstring: if ``parse_docstring`` is provided, configure
            whether to raise ValueError on invalid Google Style docstrings.
            Defaults to False.
        include_injected: Whether to include injected arguments in the schema.
            Defaults to True, since we want to include them in the schema
            when *validating* tool inputs.

    Returns:
        A pydantic model with the same arguments as the function.
    """
    sig = inspect.signature(func)
    fields = {}

    for name, param in sig.parameters.items():
        if filter_args and name in filter_args:
            continue

        # Set default or Required type in the schema
        if param.default is param.empty:
            default = ...
        else:
            default = param.default

        annotation = param.annotation if param.annotation is not param.empty else Any
        fields[name] = (annotation, default)
    model = create_model(model_name, **fields)
    if func.__qualname__ and "." in func.__qualname__:
        # Then it likely belongs in a class namespace
        in_class = True
    else:
        in_class = False

    has_args = False
    has_kwargs = False

    for param in sig.parameters.values():
        if param.kind == param.VAR_POSITIONAL:
            has_args = True
        elif param.kind == param.VAR_KEYWORD:
            has_kwargs = True

    inferred_model = model
    FILTERED_ARGS = ("run_manager", "callbacks")

    if filter_args:
        filter_args_ = filter_args
    else:
        # Handle classmethods and instance methods
        existing_params: list[str] = list(sig.parameters.keys())
        if existing_params and existing_params[0] in ("self", "cls") and in_class:
            filter_args_ = [existing_params[0]] + list(FILTERED_ARGS)
        else:
            filter_args_ = list(FILTERED_ARGS)

        for existing_param in existing_params:
            if not include_injected and _is_injected_arg_type(
                sig.parameters[existing_param].annotation
            ):
                filter_args_.append(existing_param)  # type: ignore

    description, arg_descriptions = _infer_arg_descriptions(
        func,
        parse_docstring=parse_docstring,
        error_on_invalid_docstring=error_on_invalid_docstring,
    )
    # Pydantic adds placeholder virtual fields we need to strip
    valid_properties = []
    for field in get_fields(inferred_model):
        if not has_args:
            if field == "args":
                continue
        if not has_kwargs:
            if field == "kwargs":
                continue

        if field == "v__duplicate_kwargs":  # Internal pydantic field
            continue

        if field not in filter_args_:
            valid_properties.append(field)

    return _create_subset_model_v2(
        model_name,
        inferred_model,
        list(valid_properties),
        descriptions=arg_descriptions,
        fn_description=description,
    )


class _SchemaConfig:
    """Configuration for the pydantic model.

    This is used to configure the pydantic model created from
    a function's signature.

    Parameters:
        extra: Whether to allow extra fields in the model.
        arbitrary_types_allowed: Whether to allow arbitrary types in the model.
            Defaults to True.
    """

    extra: str = "forbid"
    arbitrary_types_allowed: bool = True


@overload
def get_fields(model: type[BaseModel]) -> dict[str, FieldInfo]: ...


@overload
def get_fields(model: BaseModel) -> dict[str, FieldInfo]: ...


def get_fields(
    model: Union[
        BaseModel,
        type[BaseModel],
    ],
) -> dict[str, FieldInfo]:
    """Get the field names of a Pydantic model."""
    if hasattr(model, "model_fields"):
        return model.model_fields  # type: ignore

    elif hasattr(model, "__fields__"):
        return model.__fields__  # type: ignore
    else:
        raise TypeError(f"Expected a Pydantic model. Got {type(model)}")


class InjectedToolArg:
    """Annotation for a Tool arg that is **not** meant to be generated by a model."""


def _is_injected_arg_type(type_: type) -> bool:
    return any(
        isinstance(arg, InjectedToolArg)
        or (isinstance(arg, type) and issubclass(arg, InjectedToolArg))
        for arg in get_args(type_)[1:]
    )


def _infer_arg_descriptions(
    fn: Callable,
    *,
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = False,
) -> tuple[str, dict]:
    """Infer argument descriptions from a function's docstring."""
    if hasattr(inspect, "get_annotations"):
        # This is for python < 3.10
        annotations = inspect.get_annotations(fn)  # type: ignore
    else:
        annotations = getattr(fn, "__annotations__", {})
    if parse_docstring:
        description, arg_descriptions = _parse_python_function_docstring(
            fn, annotations, error_on_invalid_docstring=error_on_invalid_docstring
        )
    else:
        description = inspect.getdoc(fn) or ""
        arg_descriptions = {}
    if parse_docstring:
        _validate_docstring_args_against_annotations(arg_descriptions, annotations)
    for arg, arg_type in annotations.items():
        if arg in arg_descriptions:
            continue
        if desc := _get_annotation_description(arg_type):
            arg_descriptions[arg] = desc
    return description, arg_descriptions


def _parse_python_function_docstring(
    function: Callable, annotations: dict, error_on_invalid_docstring: bool = False
) -> tuple[str, dict]:
    """Parse the function and argument descriptions from the docstring of a function.

    Assumes the function docstring follows Google Python style guide.
    """
    docstring = inspect.getdoc(function)
    return _parse_google_docstring(
        docstring,
        list(annotations),
        error_on_invalid_docstring=error_on_invalid_docstring,
    )


def _validate_docstring_args_against_annotations(
    arg_descriptions: dict, annotations: dict
) -> None:
    """Raise error if docstring arg is not in type annotations."""
    for docstring_arg in arg_descriptions:
        if docstring_arg not in annotations:
            raise ValueError(
                f"Arg {docstring_arg} in docstring not found in function signature."
            )


def _is_annotated_type(typ: type[Any]) -> bool:
    return get_origin(typ) is Annotated


def _get_annotation_description(arg_type: type) -> str | None:
    if _is_annotated_type(arg_type):
        annotated_args = get_args(arg_type)
        for annotation in annotated_args[1:]:
            if isinstance(annotation, str):
                return annotation
    return None


def _parse_google_docstring(
    docstring: Optional[str],
    args: list[str],
    *,
    error_on_invalid_docstring: bool = False,
) -> tuple[str, dict]:
    """Parse the function and argument descriptions from the docstring of a function.

    Assumes the function docstring follows Google Python style guide.
    """
    if docstring:
        docstring_blocks = docstring.split("\n\n")
        if error_on_invalid_docstring:
            filtered_annotations = {
                arg for arg in args if arg not in ("run_manager", "callbacks", "return")
            }
            if filtered_annotations and (
                len(docstring_blocks) < 2 or not docstring_blocks[1].startswith("Args:")
            ):
                raise ValueError("Found invalid Google-Style docstring.")
        descriptors = []
        args_block = None
        past_descriptors = False
        for block in docstring_blocks:
            if block.startswith("Args:"):
                args_block = block
                break
            elif block.startswith("Returns:") or block.startswith("Example:"):
                # Don't break in case Args come after
                past_descriptors = True
            elif not past_descriptors:
                descriptors.append(block)
            else:
                continue
        description = " ".join(descriptors)
    else:
        if error_on_invalid_docstring:
            raise ValueError("Found invalid Google-Style docstring.")
        description = ""
        args_block = None
    arg_descriptions = {}
    if args_block:
        arg = None
        for line in args_block.split("\n")[1:]:
            if ":" in line:
                arg, desc = line.split(":", maxsplit=1)
                arg_descriptions[arg.strip()] = desc.strip()
            elif arg:
                arg_descriptions[arg.strip()] += " " + line.strip()
    return description, arg_descriptions


def _create_subset_model_v2(
    name: str,
    model: type[pydantic.BaseModel],
    field_names: list[str],
    *,
    descriptions: Optional[dict] = None,
    fn_description: Optional[str] = None,
) -> type[pydantic.BaseModel]:
    """Create a pydantic model with a subset of the model fields."""
    from pydantic import create_model
    from pydantic.fields import FieldInfo

    descriptions_ = descriptions or {}
    fields = {}
    for field_name in field_names:
        field = model.model_fields[field_name]  # type: ignore
        description = descriptions_.get(field_name, field.description)
        field_info = FieldInfo(description=description, default=field.default)
        if field.metadata:
            field_info.metadata = field.metadata
        fields[field_name] = (field.annotation, field_info)
    rtn = create_model(name, **fields)  # type: ignore

    # TODO(0.3): Determine if there is a more "pydantic" way to preserve annotations.
    # This is done to preserve __annotations__ when working with pydantic 2.x
    # and using the Annotated type with TypedDict.
    # Comment out the following line, to trigger the relevant test case.
    selected_annotations = [
        (name, annotation)
        for name, annotation in model.__annotations__.items()
        if name in field_names
    ]

    rtn.__annotations__ = dict(selected_annotations)
    rtn.__doc__ = textwrap.dedent(fn_description or model.__doc__ or "")
    return rtn

"""
Type stubs for spaCy library.

This file helps Pylance understand the interface of spaCy.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

# Core classes
class Language:
    """A spaCy NLP pipeline for processing text."""

    name: str
    pipeline: List[Tuple[str, Any]]
    meta: Dict[str, Any]

    def __init__(self, vocab: Optional[Any] = None, **kwargs: Any) -> None: ...
    def __call__(self, text: str, **kwargs: Any) -> "Doc": ...
    def disable_pipes(self, *names: str) -> "Language": ...
    def enable_pipes(self, *names: str) -> "Language": ...
    def pipe(self, texts: Iterator[str], **kwargs: Any) -> Iterator["Doc"]: ...
    def add_pipe(self, factory_name: str, **kwargs: Any) -> "Language": ...
    def remove_pipe(self, name: str) -> Tuple[str, Any]: ...

class Doc:
    """A sequence of Token objects."""

    text: str
    ents: Any
    sents: Iterator[Any]

    def __getitem__(self, key: Union[int, slice]) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...

class Vocab:
    """Storage for lexical data."""

    strings: Any

    def __init__(self, **kwargs: Any) -> None: ...
    def __getitem__(self, key: Union[int, str]) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...

# Utility functions
def load(name: str, **kwargs: Any) -> Language: ...
def blank(name: str, **kwargs: Any) -> Language: ...

class util:
    @staticmethod
    def get_data_path() -> str: ...

from . import cli

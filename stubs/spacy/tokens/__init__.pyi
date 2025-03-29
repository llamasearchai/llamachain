"""
Type stubs for spaCy tokens module.

This file helps Pylance understand the interface of spaCy's token classes.
"""
from typing import Any, Dict, List, Optional, Union, Callable, Iterator, Set, Tuple, Type, Sequence

class Doc:
    """A sequence of Token objects."""
    text: str
    ents: "Span"
    sents: Iterator["Span"]
    
    def __getitem__(self, key: Union[int, slice]) -> Union["Token", "Span"]: ...
    def __iter__(self) -> Iterator["Token"]: ...
    def __len__(self) -> int: ...
    def char_span(self, start_idx: int, end_idx: int, label: str = "", **kwargs: Any) -> Optional["Span"]: ...

class Token:
    """An individual token — i.e. a word, punctuation symbol, whitespace, etc."""
    text: str
    idx: int
    i: int
    ent_type: str
    ent_iob: str
    lemma_: str
    pos_: str
    tag_: str
    dep_: str
    head: "Token"
    
    def __init__(self, vocab: Any, doc: Doc, offset: int) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Span:
    """A slice from a Doc object."""
    doc: Doc
    start: int
    end: int
    label_: str
    text: str
    
    def __init__(self, doc: Doc, start: int, end: int, label: str = "") -> None: ...
    def __getitem__(self, key: Union[int, slice]) -> Union[Token, "Span"]: ...
    def __iter__(self) -> Iterator[Token]: ...
    def __len__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ... 
from dataclasses import dataclass
from typing import Union, List, Dict
from torch.utils.data import DataLoader


@dataclass
class Span:
    text: str
    word_start: int
    word_end: int


@dataclass
class EPDatum:
    span1: Span
    span2: Union[None, Span]
    text: str
    labels: List[str]


@dataclass
class EPDatumTokenized:
    span1: Span
    span2: Union[None, Span]
    text: str
    text_input_ids: List[int]
    text_subword_tokens: List[str]
    span1_input_ids: List[int]  # this is a bad name, what we mean is span1_subword_token_indices. But renaming this now
    # will cause all loading of tokenized instances to fail.
    span2_input_ids: Union[None, List[int]]
    span1_subword_tokens: List[str]
    span2_subword_tokens: Union[None, List[str]]
    labels: List[str]


@dataclass
class EPData:
    name: str
    metadata: str
    train: List[EPDatum]
    dev: List[EPDatum]
    test: List[EPDatum]
    labels: List[str]


@dataclass
class EPDataTokenized:
    name: str
    metadata: str
    train: List[EPDatumTokenized]
    dev: List[EPDatumTokenized]
    test: List[EPDatumTokenized]
    labels: List[str]


@dataclass
class DataLoaderOutput:
    train: DataLoader
    dev: DataLoader
    test: Dict[str, DataLoader]
    id_2_label: Dict


SPAN1 = "span1"
SPAN2 = "span2"

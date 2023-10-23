"""
create a pytorch dataloader object to be used during training.
"""
import os

import serpyco
import json
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, SequentialSampler, random_split
from dataclasses import dataclass
from eptests.core.utils import write_json_yml
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
import numpy as np
from collections import defaultdict
from transformers.tokenization_utils_base import PaddingStrategy
from eptests.datamodels import EPDatumTokenized, EPDataTokenized, DataLoaderOutput, SPAN1, SPAN2, Span
from eptests.registry import register, DATA_LOADER, DATA_FILTER_REGISTRY
from tqdm import tqdm


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    This is modified from transformers code. See the definitions there.
    """

    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    num_labels: int
    multi_label: bool
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def _multi_hot(self, batch_labels):
        _array = np.zeros((len(batch_labels), self.num_labels))
        for index, batch_label in enumerate(batch_labels):
            _array[index][batch_label] = 1
        return torch.Tensor(_array)

    @staticmethod
    def _single_hot(batch_labels):
        """
        For newer versions of PyTorch, CrossEntropyLoss works with [[1., 0., 0.], [0., 0., 1]], doesn't work the
        same with the older versions of pytorch. This is therefore a compatibility check.
        :param batch_labels:
        :return:
        """
        return torch.LongTensor([x[0] for x in batch_labels])

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        batch = self.tokenizer.pad(
            [{"input_ids": self.tokenizer.build_inputs_with_special_tokens(x["input_ids"])} for x in features],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch_mxlen = batch["input_ids"].shape[1]
        first = features[0]
        for k, v in first.items():
            if k == "input_ids":
                continue
            elif not k == "y":  # for span1 & span2
                batch[k] = torch.tensor([f[k] + [-1] * (batch_mxlen - len(f[k])) for f in features])
            else:
                if self.multi_label:
                    batch[k] = self._multi_hot([f[k] for f in features])
                else:
                    batch[k] = self._single_hot([f[k] for f in features])
        return batch


class EPDataLoader:
    def __init__(self, hf_tokenizer_model_or_loc: str, batchsz: int, label_2_id: str, multi_label: bool, **kwargs):
        self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_model_or_loc)
        self.batchsz = batchsz
        self.label_2_id_file = label_2_id
        self.ser = serpyco.Serializer(EPDataTokenized)
        self.multi_label = multi_label
        self.sampler = SequentialSampler  # TODO: change
        self.max_length = kwargs.get("max_length")
        self.filter_methods = kwargs.get("data_filter", [])

    @staticmethod
    def split_train_dev(
        train_data: List[EPDatumTokenized],
    ) -> Tuple[List[EPDatumTokenized], List[EPDatumTokenized]]:
        print("Splitting train data to train and dev")
        train_dataset, dev_dataset = random_split(
            train_data, [len(train_data) - len(train_data) // 10, len(train_data) // 10]
        )
        return train_dataset, dev_dataset

    def create_data_loader(self, data: List[EPDatumTokenized]) -> DataLoader:
        _data = self.create_data_dict(data)
        return DataLoader(
            dataset=_data,
            sampler=self.sampler(_data),
            batch_size=self.batchsz,
            collate_fn=DataCollatorWithPadding(
                tokenizer=self.hf_tokenizer,
                num_labels=len(self.label_2_id),
                multi_label=self.multi_label,
                max_length=self.max_length,
            ),
        )

    @staticmethod
    def create_ep_datum_from_json(ep_datum: Dict) -> EPDatumTokenized:
        """
        :param ep_datum:
        :return:
        """
        if ep_datum.get("span2") is not None:
            return EPDatumTokenized(
                span1=Span(**ep_datum.get("span1")),
                span2=Span(**ep_datum.get("span2")),
                text=ep_datum.get("text"),
                text_input_ids=ep_datum.get("text_input_ids"),
                text_subword_tokens=ep_datum.get("text_subword_tokens"),
                span1_input_ids=ep_datum.get("span1_input_ids"),
                span2_input_ids=ep_datum.get("span2_input_ids"),
                span1_subword_tokens=ep_datum.get("span1_subword_tokens"),
                span2_subword_tokens=ep_datum.get("span2_subword_tokens"),
                labels=ep_datum.get("labels"),
            )
        else:
            return EPDatumTokenized(
                span1=Span(**ep_datum.get("span1")),
                span2=None,
                text=ep_datum.get("text"),
                text_input_ids=ep_datum.get("text_input_ids"),
                text_subword_tokens=ep_datum.get("text_subword_tokens"),
                span1_input_ids=ep_datum.get("span1_input_ids"),
                span2_input_ids=ep_datum.get("span2_input_ids"),
                span1_subword_tokens=ep_datum.get("span1_subword_tokens"),
                span2_subword_tokens=ep_datum.get("span2_subword_tokens"),
                labels=ep_datum.get("labels"),
            )

    def load_data(self, _path: str, phases: List[str]) -> Dict:
        if _path.endswith("json"):
            data = json.load(open(_path))
            remove_phases = ({"train", "dev", "test"} - set(phases)) if phases else {}
            for remove_phase in remove_phases:
                del data[remove_phase]
            return data
        else:
            assert os.path.isdir(_path)
            _data = defaultdict(list)
            for phase in phases:
                for line in tqdm(open(f"{_path}/{phase}.jsonl"), desc=f"reading {phase} data"):
                    if line:
                        _data[phase].append(self.create_ep_datum_from_json(json.loads(line)))
            _data["labels"] = json.load(open(f"{_path}/labels.json"))
            _data["metadata"] = (
                json.load(open(f"{_path}/metadata.json")) if os.path.exists(f"{_path}/metadata.json") else None
            )
            return _data

    def __call__(self, path: str, *args, **kwargs) -> DataLoaderOutput:
        self.path = path
        print("reading data file(s)..")
        phases = kwargs.get("phases", ["train", "dev", "test"])
        ep_data = self.load_data(path, phases)
        self.label_2_id = {v: k for k, v in enumerate(ep_data["labels"])}
        self.id_2_label = {v: k for k, v in self.label_2_id.items()}
        write_json_yml(self.label_2_id, self.label_2_id_file, yml=True)
        if "test" in phases:
            test_data_filter_mapping = {"no_filter": ep_data["test"]}
            if self.filter_methods:
                for _filter_method in self.filter_methods:
                    data_filter = DATA_FILTER_REGISTRY[_filter_method]()
                    test_data_filter_mapping[_filter_method] = data_filter(ep_data["train"], ep_data["test"])
        return DataLoaderOutput(
            train=self.create_data_loader(ep_data["train"]),
            dev=self.create_data_loader(ep_data["dev"]),
            test={k: self.create_data_loader(v) for (k, v) in test_data_filter_mapping.items()}
            if "test" in phases
            else [],
            id_2_label=self.id_2_label,
        )


@register(_name="one-span", _type=DATA_LOADER)
class OneSpanEPDataLoader(EPDataLoader):
    def __init__(self, hf_tokenizer_model_or_loc: str, batchsz: int, label_2_id: str, **kwargs):
        super().__init__(
            hf_tokenizer_model_or_loc=hf_tokenizer_model_or_loc, batchsz=batchsz, label_2_id=label_2_id, **kwargs
        )

    def create_data_dict(self, data: List[EPDatumTokenized]) -> List[Dict]:
        return [
            {
                "input_ids": datum.text_input_ids,
                SPAN1: datum.span1_input_ids,
                "y": [self.label_2_id[label] for label in datum.labels],
            }
            for datum in data
        ]


@register(_name="two-span", _type=DATA_LOADER)
class TwoSpanEPDataLoader(EPDataLoader):
    def __init__(self, hf_tokenizer_model_or_loc: str, batchsz: int, label_2_id: str, **kwargs):
        super().__init__(
            hf_tokenizer_model_or_loc=hf_tokenizer_model_or_loc, batchsz=batchsz, label_2_id=label_2_id, **kwargs
        )

    def create_data_dict(self, data: List[EPDatumTokenized]) -> List[Dict]:
        return [
            {
                "input_ids": datum.text_input_ids,
                SPAN1: datum.span1_input_ids,
                SPAN2: datum.span2_input_ids,
                "y": [self.label_2_id[label] for label in datum.labels],
            }
            for datum in data
        ]

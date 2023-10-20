from eptests.datamodels import EPDatumTokenized
from typing import List
from eptests.registry import DATA_FILTER, register
from collections import defaultdict as ddict
import random
from random import choices
from collections import Counter


class FilterData:
    def __init__(self):
        random.seed(30)

    def __call__(self, train_data: List[EPDatumTokenized], test_data: List[EPDatumTokenized]) -> List[EPDatumTokenized]:
        return NotImplementedError


@register(_name="train-input", _type=DATA_FILTER)
class TrainDataRemoval(FilterData):
    """
    Remove exact input data from test data
    """

    def __call__(self, train_data: List[EPDatumTokenized], test_data: List[EPDatumTokenized]) -> List[EPDatumTokenized]:
        print("running filter: [train-input]")
        unq_train_input = set()
        filtered_test_data = []

        for data in train_data:
            if data.span2 is None:
                unq_train_input.add(frozenset((data.span1.text, tuple(sorted(data.labels)))))
            else:
                unq_train_input.add(frozenset((data.span1.text, data.span2.text, tuple(sorted(data.labels)))))
        for data in test_data:
            if data.span2 is None:
                if frozenset((data.span1.text, tuple(sorted(data.labels)))) not in unq_train_input:
                    filtered_test_data.append(data)
            else:
                if frozenset((data.span1.text, data.span2.text, tuple(sorted(data.labels)))) not in unq_train_input:
                    filtered_test_data.append(data)

        return filtered_test_data


@register(_name="random-sampling", _type=DATA_FILTER)
class RandomSample(FilterData):
    """
    Assume i/p having different labels. e.g. back as verb, noun, adjective etc in training data.
    random sample these labels and remove from test if they match. IN MLC take entire label set as one unit.
    """

    def __call__(self, train_data: List[EPDatumTokenized], test_data: List[EPDatumTokenized]) -> List[EPDatumTokenized]:
        print("running filter: [random-sampling]")
        train_to_lbls = ddict(list)
        filtered_test_data = []
        for data in train_data:
            if (
                data.span2 is not None
                and data.labels not in train_to_lbls[frozenset((data.span1.text, data.span2.text))]
            ):
                train_to_lbls[frozenset((data.span1.text, data.span2.text))].append(data.labels)
            elif data.span2 is None and data.labels not in train_to_lbls[frozenset((data.span1.text))]:
                train_to_lbls[frozenset((data.span1.text))].append(data.labels)

        for data in test_data:
            if data.span2 is not None and frozenset((data.span1.text, data.span2.text)) in train_to_lbls.keys():
                res = choices(train_to_lbls[frozenset((data.span1.text, data.span2.text))])
                if sorted(res[0]) == sorted(data.labels):
                    continue
            elif data.span2 is None and frozenset((data.span1.text)) in train_to_lbls.keys():
                res = choices(train_to_lbls[frozenset((data.span1.text))])
                if sorted(res[0]) == sorted(data.labels):
                    continue
            filtered_test_data.append(data)
        return filtered_test_data


@register(_name="train-dist-sampling", _type=DATA_FILTER)
class TrainSample(FilterData):
    """
    Assume i/p having different labels. e.g. back as verb, noun, adjective etc in training data.
    sample these labels from training data distribution and remove from test if they match,
    as long as distribution is not uniform in those labels. Take MLC as one unit.
    """

    def __call__(self, train_data: List[EPDatumTokenized], test_data: List[EPDatumTokenized]) -> List[EPDatumTokenized]:
        print("running filter: [train-dist-sampling]")
        spn_wght = {}
        label_dict = ddict(int)
        lbl_idx = 0
        for data in train_data:
            if tuple(sorted(data.labels)) not in label_dict.keys():
                label_dict[tuple(sorted(data.labels))] = lbl_idx
                lbl_idx += 1

        for data in train_data:
            if data.span2 is None:
                if data.span1.text not in spn_wght.keys():
                    spn_wght[data.span1.text] = [0 for _ in label_dict.keys()]
                spn_wght[data.span1.text][label_dict[tuple(sorted(data.labels))]] += 1
            else:
                if frozenset((data.span1.text, data.span2.text)) not in spn_wght.keys():
                    spn_wght[frozenset((data.span1.text, data.span2.text))] = [0 for _ in label_dict.keys()]
                spn_wght[frozenset((data.span1.text, data.span2.text))][label_dict[tuple(sorted(data.labels))]] += 1

        filtered_test_data = []

        for data in test_data:
            if data.span2 is not None and frozenset((data.span1.text, data.span2.text)) in spn_wght.keys():
                _lbl_wgt = spn_wght[frozenset((data.span1.text, data.span2.text))]
                # print("len(_lbl_wgt)",len(_lbl_wgt))
                # print("len(label_dict.keys())",len(label_dict.keys()))
                count = Counter(_lbl_wgt)
                # print(_lbl_wgt)
                # print(label_dict)
                if len(count.keys()) > 2 or (len(count.keys()) == 2 and 0 not in count.keys()):
                    res = choices([i for i, _ in enumerate(label_dict.keys())], _lbl_wgt)
                    if (
                        tuple(sorted(data.labels)) in label_dict.keys()
                        and res[0] == label_dict[tuple(sorted(data.labels))]
                    ):
                        continue
            elif data.span2 is None and data.span1.text in spn_wght.keys():
                _lbl_wgt = spn_wght[data.span1.text]
                count = Counter(_lbl_wgt)
                if len(count.keys()) > 2 or (len(count.keys()) == 2 and 0 not in count.keys()):
                    res = choices([i for i, _ in enumerate(label_dict.keys())], _lbl_wgt)
                    if (
                        tuple(sorted(data.labels)) in label_dict.keys()
                        and res[0] == label_dict[tuple(sorted(data.labels))]
                    ):
                        continue
            filtered_test_data.append(data)
        return filtered_test_data

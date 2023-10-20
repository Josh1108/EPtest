import json
import os
from abc import ABC
from typing import List, Tuple
from eptests.datamodels import EPDatum, EPData, Span
from conllu.parser import parse_line, DEFAULT_FIELDS
import random
import itertools


class Converter:
    """
    Base Converter class used to run all conversions
    """

    def __init__(
        self,
        dir_path: str,
        name: str,
        train_file: str,
        test_file: str,
        dev_file: str = "",
        metadata: str = "",
        dev_set: bool = True,
    ):

        self.dir_path = dir_path
        self.dev_set = dev_set
        self.name = name
        self.metadata = metadata
        self.train_file = train_file
        self.test_file = test_file
        self.dev_file = dev_file

    def run(self) -> EPData:
        """
        Runs conversions over train, test and dev ( optional) files

        """

        dev, train, test = [], [], []  # list of EPDatum

        labels = set()

        if self.dev_set:
            dev = self.run_file(os.path.join(self.dir_path, self.dev_file))
        train = self.run_file(os.path.join(self.dir_path, self.train_file))
        test = self.run_file(os.path.join(self.dir_path, self.test_file))

        for item in train:
            for lbl in item.labels:
                labels.add(lbl)
        # check if label in train set, else drop.
        dr_item = 0
        test_filt = []
        dev_filt = []

        for item in test:
            for lbl in item.labels:
                if lbl not in labels:
                    dr_item += 1
                else:
                    test_filt.append(item)
        print(f"{dr_item} rows are dropped from test data due to oov labels")

        if self.dev_set:
            dr_item = 0
            for item in dev:
                for lbl in item.labels:
                    if lbl not in labels:
                        dr_item += 1
                    else:
                        dev_filt.append(item)
            print(f"{dr_item} rows are dropped from dev data due to oov labels")

        dataset = EPData(self.name, self.metadata, train, dev_filt, test_filt, list(labels))
        return dataset

    def run_file(self, file_path: str) -> List[EPDatum]:
        """
        code for conversion for each file to required format
        """
        raise NotImplementedError


class BaseOneSpan(Converter):
    """

    Base class for {<span>} probing tasks. It inherits class `Converter`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_file(self, file_path: str) -> List[EPDatum]:

        list_data = []  # list to store all data points

        with open(file_path, "r") as f:
            json_list = list(f)

        for json_str in json_list:

            dataitem = json.loads(json_str)
            txt = dataitem["text"]
            for item in dataitem["targets"]:

                spn1_txt = " ".join(txt.split(" ")[item["span1"][0] : item["span1"][1]])
                s1 = Span(spn1_txt, item["span1"][0], item["span1"][1])
                s2 = None
                labels = [item["label"]]
                list_data.append(EPDatum(s1, s2, txt, labels))

        return list_data


class BaseTwoSpan(Converter):
    """
    Base class for {<span>,<span>} probing tasks. It inherits class `Converter`

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_file(self, file_path: str) -> List[EPDatum]:
        list_data = []  # list to store all data points

        with open(file_path, "r") as f:
            json_list = list(f)

        for json_str in json_list:

            dataitem = json.loads(json_str)
            txt = dataitem["text"]
            for item in dataitem["targets"]:
                spn1_txt = " ".join(txt.split(" ")[item["span1"][0] : item["span1"][1]])
                spn2_txt = " ".join(txt.split(" ")[item["span2"][0] : item["span2"][1]])
                s_1 = Span(spn1_txt, item["span1"][0], item["span1"][1])
                s_2 = Span(spn2_txt, item["span2"][0], item["span2"][1])
                labels = [item["label"]]
                list_data.append(EPDatum(s_1, s_2, txt, labels))

        return list_data


class OntonotesConstituent(BaseOneSpan):
    pass


class OntonotesPos(BaseOneSpan):
    pass


class OntonotesSrl(BaseTwoSpan):
    pass


class OntonotesNer(BaseOneSpan):
    pass


class Dpr(BaseTwoSpan):
    pass


class DepEwt(BaseTwoSpan):
    pass


class OntonotesCoref(BaseTwoSpan):
    pass


class Rel(BaseTwoSpan):
    pass


class Conll2000Chunking(BaseOneSpan):
    """

    Reads a file in a format where each line is:
    word<space>pos<space>tag
    and sentences are separated by newlines. The POS info is discarded.
    This DatasetReader is used to process the CoNLL chunking data.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.FIELDS = ["form", "pos", "tag"]

    def parse_sentence(self, sentence: str):
        annotated_sentence = []

        lines = [line for line in sentence.split("\n") if line]

        for line_idx, line in enumerate(lines):
            annotated_token = dict(zip(self.FIELDS, line.split(" ")))
            annotated_sentence.append(annotated_token)
        return annotated_sentence

    def parse(self, text: str):
        data = []
        for sent in text.split("\n\n"):
            if sent:
                data.append(self.parse_sentence(sent))
        return data

    def run_file(self, file_path: str) -> List[EPDatum]:
        list_data = []  # list to store all data points

        with open(file_path, "r") as f:
            data = f.read()

        lis_sent = self.parse(data)
        for lis_item in lis_sent:
            txt = ""
            for dicti in lis_item[:-1]:
                txt += dicti["form"] + " "
            txt += lis_item[-1]["form"]

            for i, dicti in enumerate(lis_item):
                s1 = Span(dicti["form"], i, i + 1)
                s2 = None
                labels = [dicti["tag"]]
                list_data.append(EPDatum(s1, s2, txt, labels))

        return list_data


class ConlluPOS(BaseOneSpan):
    """
    Reads a file in the conllu Universal Dependencies format and returns
    instances suitable for POS tagging.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse(self, text: str, fields: Tuple = DEFAULT_FIELDS) -> list:
        data = []
        for sentence in text.split("\n\n"):
            if sentence:
                data.append(
                    [
                        parse_line(line, fields)
                        for line in sentence.split("\n")
                        if line and not line.strip().startswith("#")
                    ]
                )
        return data

    def run_file(self, file_path: str) -> List[EPDatum]:
        list_data = []
        with open(file_path, "r") as f:
            data = f.read()
        lis_sent = self.parse(data)
        for lis_item in lis_sent:
            txt = ""
            for dicti in lis_item[:-1]:
                txt += dicti["form"] + " "
            txt += lis_item[-1]["form"]
            for i, dicti in enumerate(lis_item):
                s_1 = Span(dicti["form"], i, i + 1)
                s_2 = None
                labels = [dicti["upostag"]]
                list_data.append(EPDatum(s_1, s_2, txt, labels))

        return list_data


class Spr2(Converter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_file(self, file_path: str) -> List[EPDatum]:
        list_data = []  # list to store all data points

        with open(file_path, "r") as f:
            json_list = list(f)

        for json_str in json_list:

            dataitem = json.loads(json_str)
            txt = dataitem["text"]
            for item in dataitem["targets"]:
                s1 = Span(item["info"]["span1_text"], item["span1"][0], item["span1"][1])
                s2 = Span(item["info"]["span2_txt"], item["span2"][0], item["span2"][1])
                labels = item["label"]
                list_data.append(EPDatum(s1, s2, txt, labels))

        return list_data


class Spr1(Spr2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_file(self, file_path: str) -> List[EPDatum]:
        list_data = []  # list to store all data points

        with open(file_path, "r") as f:
            json_list = list(f)

        for json_str in json_list:

            dataitem = json.loads(json_str)
            txt = dataitem["text"]
            for item in dataitem["targets"]:
                spn1_txt = " ".join(txt.split(" ")[item["span1"][0] : item["span1"][1]])
                spn2_txt = " ".join(txt.split(" ")[item["span2"][0] : item["span2"][1]])

                s1 = Span(spn1_txt, item["span1"][0], item["span1"][1])
                s2 = Span(spn2_txt, item["span2"][0], item["span2"][1])

                labels = item["label"]
                list_data.append(EPDatum(s1, s2, txt, labels))

        return list_data


class PosPenn(BaseOneSpan):
    """
    run this against the conllx files generated by Treebank preprocessing used
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.FIELDS = ["id", "form", "lemma", "coarse_pos", "pos"]

    def parse(self, text: str) -> list:
        data = []
        for sentence in text.split("\n\n"):
            if sentence:
                annotated_sent = []

                lines = [line for line in sentence.split("\n") if line and not line.strip().startswith("#")]

                for line_idx, line in enumerate(lines):
                    annotated_token = dict(zip(self.FIELDS, line.split("\t")))
                    annotated_sent.append(annotated_token)
                data.append(annotated_sent)

        return data

    def run_file(self, file_path: str) -> List[EPDatum]:

        list_data = []
        with open(file_path, "r") as f:
            data = f.read()

        lis_sent = self.parse(data)

        for lis_item in lis_sent:
            txt = ""
            for dicti in lis_item[:-1]:
                txt += dicti["form"] + " "
            txt += lis_item[-1]["form"]

            for i, dicti in enumerate(lis_item):
                s_1 = Span(dicti["form"], i, i + 1)
                s_2 = None
                labels = [dicti["pos"]]
                list_data.append(EPDatum(s_1, s_2, txt, labels))

        return list_data


class SyntacticDepenedencyPredictionEwt(Converter):
    def _sample_negative_indices(
        self, child_index: int, all_arc_indices: List[Tuple[int, int]], seq_len: int  # type: ignore
    ):
        """
        Given a child index, generate a new (child, index) pair where
        ``index`` refers to an index in the sequence that is not a parent
        of the provided child.
        Parameters
        ----------
        child_index: ``int``, required
            The index of the child to sample a negative example for.
        all_arc_indices: ``List[Tuple[int, int]]``, required
            A list of (child index, parent index) pairs that correspond to each
            dependency arc in the sentence. Indices are 0 indexed.
        seq_len: ``int``, required
            The length of the sequence.
        """
        # All possible parents
        all_possible_parents = set(range(seq_len))
        # Find all the parents of the child.
        parents = {arc_indices[1] for arc_indices in all_arc_indices if arc_indices[0] == child_index}
        # Get the indices that are not parents of the child or the child itself
        non_parents = all_possible_parents - parents.union({child_index})
        # If there are no indices that are not a parent.
        if not non_parents:
            return None
        return (child_index, random.Random(0).sample(non_parents, 1)[0])

    def parse(self, text: str, fields: Tuple = DEFAULT_FIELDS) -> list:
        data = []
        for sentence in text.split("\n\n"):
            if sentence:
                annotations = [
                    parse_line(line, fields)
                    for line in sentence.split("\n")
                    if line and not line.strip().startswith("#")
                ]
                # (child, parent/head) pairs
                arc_indices = []
                # Strings with the relation for each pair
                arc_labels = []
                for idx, annotation in enumerate(annotations):
                    head = annotation["head"]
                    if head == 0 or head is None:
                        # Skip the root
                        continue
                    # UD marks the head with 1-indexed numbering, so we subtract
                    # one to get the index of the parent.
                    arc_indices.append((idx, head - 1))
                    arc_labels.append(annotation["deprel"])
                data.append((annotations, arc_indices, arc_labels))
        return data

    def run_file(self, file_path: str) -> List[EPDatum]:

        list_data = []
        with open(file_path, "r") as f:
            data = f.read()

        lis_sent = self.parse(data)

        for annotation, directed_arc_indices, arc_labels in lis_sent:
            tokens = [x["form"] for x in annotation]
            txt = " ".join(tokens)
            if not directed_arc_indices:
                continue

            for i, (a, b) in enumerate(directed_arc_indices):

                s_1 = Span(annotation[a]["form"], a, a + 1)
                s_2 = Span(annotation[b]["form"], b, b + 1)
                labels = ["1"]  # true
                list_data.append(EPDatum(s_1, s_2, txt, labels))
                negative_arc_index = self._sample_negative_indices(
                    child_index=a, all_arc_indices=directed_arc_indices, seq_len=len(tokens)
                )
                # print(a, negative_arc_index)
                if negative_arc_index:
                    s_1 = Span(annotation[a]["form"], a, a + 1)
                    s_2 = Span(
                        annotation[negative_arc_index[1]]["form"],
                        negative_arc_index[1],
                        negative_arc_index[1] + 1,
                    )
                    labels = ["0"]
                    list_data.append(EPDatum(s_1, s_2, txt, labels))

        return list_data


class SyntacticDependencyPredictionPenn(SyntacticDepenedencyPredictionEwt):
    pass


class SyntacticDependencyClasssificationEwt(Converter):
    def parse(self, text: str, fields: Tuple = DEFAULT_FIELDS) -> list:
        data = []
        for sentence in text.split("\n\n"):
            if sentence:
                annotations = [
                    parse_line(line, fields)
                    for line in sentence.split("\n")
                    if line and not line.strip().startswith("#")
                ]
                # (child, parent/head) pairs
                arc_indices = []
                # Strings with the relation for each pair
                arc_labels = []
                for idx, annotation in enumerate(annotations):
                    head = annotation["head"]
                    if head == 0 or head is None:
                        # Skip the root
                        continue
                    # UD marks the head with 1-indexed numbering, so we subtract
                    # one to get the index of the parent.
                    arc_indices.append((idx, head - 1))
                    arc_labels.append(annotation["deprel"])
                data.append((annotations, arc_indices, arc_labels))
        return data

    def run_file(self, file_path: str) -> List[EPDatum]:
        list_data = []
        with open(file_path, "r") as f:
            data = f.read()

        lis_sent = self.parse(data)

        for annotation, directed_arc_indices, arc_labels in lis_sent:
            tokens = [x["form"] for x in annotation]
            txt = " ".join(tokens)
            if not directed_arc_indices:
                continue

            for i, (a, b) in enumerate(directed_arc_indices):

                s_1 = Span(annotation[a]["form"], a, a + 1)
                s_2 = Span(annotation[b]["form"], b, b + 1)
                labels = arc_labels[i]  # true
                list_data.append(EPDatum(s_1, s_2, txt, [labels]))
        return list_data


class SyntacticDependencyClassificationPenn(SyntacticDependencyClasssificationEwt):
    pass



class Conll2003NER(Converter):
    """
    Reads instances from a pretokenised file where each line is in the following format:
    WORD POS-TAG CHUNK-TAG NER-TAG
    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.
    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent ``Instance``. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _is_divider(line: str) -> bool:
        empty_line = line.strip() == ""
        if empty_line:
            return True
        else:
            first_token = line.split()[0]
            if first_token == "-DOCSTART-":
                return True
            else:
                return False

    def run_file(self, file_path: str) -> List[EPDatum]:

        lis_data = []
        with open(file_path, "r") as data_file:
            for is_divider, lines in itertools.groupby(data_file, self._is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    tokens, _, _, ner_tags = [list(field) for field in zip(*fields)]
                    txt = " ".join(tokens)

                    for i, (token, ner_tag) in enumerate(zip(tokens, ner_tags)):
                        s_1 = Span(token, i, i + 1)
                        s_2 = None
                        labels = [ner_tag]
                        lis_data.append(EPDatum(s_1, s_2, txt, labels))
        return lis_data

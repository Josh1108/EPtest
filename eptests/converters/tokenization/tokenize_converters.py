import json
import pickle
from typing import List
from eptests.datamodels import EPDatum, EPDatumTokenized, EPDataTokenized
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import serpyco
from pathlib import Path
from spacy_alignments import get_alignments

ser = serpyco.Serializer(EPDataTokenized)


class SubWordTokenizer:
    def __init__(self, hf_model_name: str, model_special_tokens: bool):
        self.hf_model_name = hf_model_name
        self.model_special_tokens = model_special_tokens

    def load_dataset(self, path: str):
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        return

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name, use_fast=True)
        return

    def tokenize(self, ep_datum: EPDatum) -> EPDatumTokenized:
        raise NotImplementedError

    def align(self, other_tokens: List[str]) -> List[List[int]]:
        """
        the returned list has the same length as the other_tokens list
        :param other_tokens:
        :return:
        """
        raise NotImplementedError


class BPE(SubWordTokenizer):
    """
    Although this is defined with BPE in mind, this implementation works for all tokenizations
    barring SentencePiece, for which mapping tokens and words is not feasible
    """



    def lazy_batch_tokenizer(self, ep_data: List[EPDatum], batch_size=10):
        text_to_tokenize = []
        for item in ep_data:
            text_to_tokenize.append(item.text)
        rem = len(text_to_tokenize) % batch_size
        if rem > 0:
            rem = 1
        else:
            rem = 0

        for i in range(0, (len(text_to_tokenize) // batch_size) + rem):
            yield self.tokenizer(text_to_tokenize[i * batch_size : min((i + 1) * batch_size, len(text_to_tokenize))])

    @staticmethod
    def remove_special_tokens(lis: List) -> List:
        return lis[1 : len(lis) - 1]

    def batch_tokenize_ep(self, ep_data: List[EPDatum], batch_size: int = 10):
        all_ep_datum_tokenized = []
        count_of_unk_sentences = 0
        btokenizer = self.lazy_batch_tokenizer(ep_data=ep_data, batch_size=batch_size)

        for i, encoded in tqdm(enumerate(btokenizer)):
            for j in range(len(encoded["input_ids"])):

                text_input_ids = encoded["input_ids"][j]  # this includes [cls] and [sep]
                text_subword_tokens = encoded.tokens(j)  # includes [cls] & [sep]
                whitespace_tokens = ep_data[batch_size * i + j].text.split(" ")

                if self.model_special_tokens:
                    text_input_ids = self.remove_special_tokens(text_input_ids)
                    text_subword_tokens = self.remove_special_tokens(text_subword_tokens)
                repl_text_subword_tokens = text_subword_tokens
                if self.hf_model_name == "roberta-base":
                    repl_text_subword_tokens = [x.replace("Ġ", "") for x in text_subword_tokens]
                whitespace_to_tx, tx_to_whitespace = get_alignments(whitespace_tokens, repl_text_subword_tokens)
                if [] in whitespace_to_tx:
                    count_of_unk_sentences += 1
                    continue  # removing unk token
                tok_to_words = []
                for t, item in enumerate(whitespace_to_tx):
                    for p in range(len(item)):
                        tok_to_words.append(t)

                word1_id = [
                    k
                    for k in range(
                        ep_data[batch_size * i + j].span1.word_start,
                        ep_data[batch_size * i + j].span1.word_end,
                    )
                ]  # like [0,1]
                span1_ids = [k for k, x in enumerate(tok_to_words) if x in word1_id]
                # span1_input_ids = [text_input_ids[x] for x in span1_ids]
                span1_subword_tokens = [text_subword_tokens[x] for x in span1_ids]
                if ep_data[batch_size * i + j].span2 is None:
                    span2_ids = None
                    span2_subword_tokens = None
                else:
                    word2_id = [
                        i
                        for i in range(
                            ep_data[batch_size * i + j].span2.word_start,
                            ep_data[batch_size * i + j].span2.word_end,
                        )
                    ]  # like [0,1]
                    span2_ids = [i for i, x in enumerate(tok_to_words) if x in word2_id]
                    # span2_input_ids = [text_input_ids[x] for x in span2_ids]
                    span2_subword_tokens = [text_subword_tokens[x] for x in span2_ids]

                ep_datum_tokenized = EPDatumTokenized(
                    span1=ep_data[batch_size * i + j].span1,
                    span2=ep_data[batch_size * i + j].span2,
                    text=ep_data[batch_size * i + j].text,
                    text_input_ids=text_input_ids,
                    text_subword_tokens=text_subword_tokens,
                    span1_input_ids=span1_ids,
                    span1_subword_tokens=span1_subword_tokens,
                    span2_subword_tokens=span2_subword_tokens,
                    span2_input_ids=span2_ids,
                    labels=ep_data[batch_size * i + j].labels,
                )
                all_ep_datum_tokenized.append(ep_datum_tokenized)
                # print(ep_datum_tokenized)
        print(count_of_unk_sentences)
        return all_ep_datum_tokenized

    def save_dataset(self, path: str):
        data = self.ep_data_tokenized
        with open(path, "w") as wf:
            wf.write(json.dumps(ser.dump(data)))

    def run_all(self, batch_size=10):
        ep_data = self.data

        train_tokenized_ep_datum = self.batch_tokenize_ep(ep_data.train, batch_size)
        test_tokenized_ep_datum = self.batch_tokenize_ep(ep_data.test, batch_size)
        dev_tokenized_ep_datum = []
        if ep_data.dev:
            dev_tokenized_ep_datum = self.batch_tokenize_ep(ep_data.dev, batch_size)

        self.ep_data_tokenized = EPDataTokenized(
            name=ep_data.name,
            metadata=ep_data.metadata,
            train=train_tokenized_ep_datum,
            dev=dev_tokenized_ep_datum,
            test=test_tokenized_ep_datum,
            labels=ep_data.labels,
        )
        return self.ep_data_tokenized


if __name__ == "__main__":

    # Listing all datasets we have

    dir = "/home/jushank/eptests/data/EPDatum-datasets"

    save_dir = "/home/jushank/eptests/data/EPDatumTokenized-datasets"

    files_lis = []
    completed_files = []
    list_models = ["bert-base-cased", "roberta-base"]
    model_special_tokens = [True, True]

    batch_size = 10
    for path in os.listdir(dir):
        full_path = os.path.join(dir, path)
        files_lis.append(full_path)

    for i, model in enumerate(list_models):
        conv = BPE(hf_model_name=model, model_special_tokens=model_special_tokens[i])
        conv.load_tokenizer()

        for file in files_lis:
            if file.split(".pickle")[0].split("/")[-1] + ".json" in os.listdir(os.path.join(save_dir, model)):
                continue
            print(f"======{file}=====")
            conv.load_dataset(file)
            converted_data = conv.run_all(batch_size)
            Path(os.path.join(save_dir, model)).mkdir(parents=True, exist_ok=True)
            conv.save_dataset(os.path.join(save_dir, model, file.split(".pickle")[0].split("/")[-1] + ".json"))

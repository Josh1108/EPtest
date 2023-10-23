"""
This script will create config files for single label multi class classification tasks, following ewt-pos
"""
import argparse
from eptests.core.utils import read_json_or_yml, write_json_yml
import os
import json
from typing import Dict

PHASES = ["train", "dev", "test"]
REPLACEMENTS = [(os.environ["EP_TESTS_DATA"], "${EP_TESTS_DATA}"), (os.environ["EP_TESTS_HOME"], "${EP_TESTS_HOME}")]


def calculate_max_length(data_: Dict) -> int:
    max_lengths = [max([len(x["text_input_ids"]) for x in data_[phase]]) for phase in PHASES]
    return max(max_lengths)


def is_one_or_two_span(data_: Dict) -> str:
    if data_["train"][0]["span2"] is not None:
        return "two-span"
    return "one-span"


def is_multi_label(data_: Dict) -> bool:
    _num_labels = [set([len(x["labels"]) for x in data_[phase]]) for phase in PHASES]
    if frozenset().union(*_num_labels) == {1}:
        return False
    return True


def create_label_2_id(data_: Dict, label_2_id_path_: str) -> int:
    labels = {label: index for index, label in enumerate(data_["labels"])}
    write_json_yml(labels, path=label_2_id_path_, yml=True)
    return len(labels)


def get_warmup_steps(data_: Dict, num_epochs: int, batchsz: int, warm_up_ratio: float = 0.1) -> int:
    """
    total number of steps = data size * number of epochs / batch size
    :return:
    """
    num_steps = (len(data_["train"]) * num_epochs) / batchsz
    return int(num_steps * warm_up_ratio)


parser = argparse.ArgumentParser("create config file from data")
parser.add_argument("--new_task", default="conll-2000-chunking")
parser.add_argument("--model", default="bertbc")
parser.add_argument("--train_epochs", default=3)
parser.add_argument("--batchsz", default=80)
args = parser.parse_args()

model_hf_name_map = {"bertbc": "bert-base-cased", "rb": "roberta-base"}

base_task = "ewt-pos"
base_file = os.path.expanduser(f"~/dev/work/eptests/eptests/core/config/{base_task}-bertbc.yml")
new_task = args.new_task
output_file = base_file.replace(base_task, new_task).replace("bertbc", args.model)
model_hf_name = model_hf_name_map[args.model]

config = read_json_or_yml(base_file)
config["dataset"]["name"] = new_task
config["dataset"]["path"] = (
    config["dataset"]["path"].replace(base_task, new_task).replace("bert-base-cased", model_hf_name)
)
if args.use_dir_path:
    config["dataset"]["path"] = config["dataset"]["path"][:-5]

print(f"loading dataset for task: {new_task}")
data = json.load(open(config["dataset"]["path"]))

one_or_two_span = is_one_or_two_span(data)
print(f"one or two span: {one_or_two_span}")
multi_label = is_multi_label(data)
print(f"multi label: {multi_label}")
max_length = calculate_max_length(data)
print(f"max length: {max_length}")
label_2_id_path = config["loader"]["label_2_id"].replace(base_task, new_task)
num_labels = create_label_2_id(data, label_2_id_path)
print(f"label 2 id written, num_labels {num_labels}")

config["loader"]["name"] = one_or_two_span
config["loader"]["label_2_id"] = label_2_id_path
config["loader"]["multi_label"] = multi_label
config["loader"]["max_length"] = max_length
config["loader"]["hf_tokenizer_model_or_loc"] = model_hf_name
config["loader"]["batchsz"] = args.batchsz

config["model"]["probe"]["name"] = one_or_two_span
config["model"]["probe"]["multi_label"] = multi_label
config["model"]["classifier"]["num_labels"] = num_labels
config["model"]["encoder"]["hf_model_or_loc"] = model_hf_name

config["train"]["epochs"] = args.train_epochs
warmup_steps = get_warmup_steps(data, config["train"]["epochs"], config["loader"]["batchsz"])
print(f"warmup steps: {warmup_steps}")
config["train"]["warmup_steps"] = warmup_steps

config["label"] = f"{new_task}-{args.model}"

write_json_yml(config, output_file, yml=True, replacements=REPLACEMENTS)
print(f"new config written to {output_file}")

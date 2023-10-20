from typing import Dict, List
from jiant.utils.python.io import read_json_or_yml
from baseline.pytorch.classify.model import ClassifierModelBase
from baseline.pytorch.classify.train import fit
from baseline.reporting import create_reporting
from mead.tasks import merge_reporting_with_settings

# flake8: noqa
from eptests.core.models import *
from eptests.registry import DATA_LOADER_REGISTRY, PROBE_REGISTRY
from eptests.datamodels import DataLoaderOutput
from eptests.core.utils import remove_key_from_dict, get_device

import argparse


def get_data(loader_params: Dict, data_path: str) -> DataLoaderOutput:
    data_loader = DATA_LOADER_REGISTRY[loader_params["name"]](**remove_key_from_dict(loader_params))
    return data_loader(data_path)


def create_model(model_params: Dict, id_2_label: Dict) -> ClassifierModelBase:
    probe_params = model_params["probe"]
    del model_params["probe"]
    probe = PROBE_REGISTRY[probe_params["name"]](**{**remove_key_from_dict(probe_params), **model_params})
    probe.labels = id_2_label
    return probe


def create_reporters(reporting: List[Dict], config_file: str):
    reporting_hooks, reporting = merge_reporting_with_settings(reporting, {})
    return create_reporting(reporting_hooks, reporting, {"config_file": config_file, "task": "eptest", "base_dir": "."})


def main():
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--config_file", default="config/spr2.yml")
    args = parser.parse_args()
    exp_params = read_json_or_yml(args.config_file)

    data = get_data(exp_params["loader"], exp_params["dataset"]["path"])
    print("data loaded, loading model")

    model = create_model(exp_params["model"], data.id_2_label)
    print(get_device())
    if get_device().startswith("cuda"):
        model.cuda(get_device())
    print("model loaded, loading reporters")

    reporters = create_reporters(exp_params["reporting"], config_file=args.config_file)
    print("reporters loaded, training")

    train_params = exp_params["train"]
    train_params["reporting"] = [x.step for x in reporters]
    fit(model, ts=data.train, vs=data.dev, es=data.test["no_filter"], **train_params)
    for reporter in reporters:
        reporter.done()


if __name__ == "__main__":
    main()

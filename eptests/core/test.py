"""
Test a trained model, with dataset filtering
"""
from jiant.utils.python.io import read_json_or_yml
from eptests.core.trainer import get_data
from baseline.train import create_trainer
from typing import Dict, Union, List
from eptests.core.modelfinder import BestModelFinderWandb
from eptests.core.utils import get_device
import torch
import os
import shutil
import argparse
from mead.tasks import merge_reporting_with_settings
from baseline.reporting import create_reporting

# flake8: noqa
from eptests.core.customtrainer import mlc


def create_test_reporters(reporting: List[Dict], config_file: str, filt: str):
    reporting_hooks, reporting = merge_reporting_with_settings(reporting, {})
    return create_reporting(
        reporting_hooks, reporting, {"config_file": config_file, "task": "eptest", "base_dir": ".", "filt": filt}
    )


def get_model(model_loc_or_config: Union[str, Dict]) -> str:
    """
    model_loc_or config is either a string, in which case, it is the direct path to the model to be loaded. Else,
    it has params to get a model from wandb.
    :param model_loc_or_config:
    :return:
    """
    if type(model_loc_or_config) == str:
        return model_loc_or_config
    else:
        model_finder = BestModelFinderWandb(dataset=model_loc_or_config["dataset"], label=model_loc_or_config["label"])
        best_model_info = model_finder.run(metric=model_loc_or_config["metric"])
        if any([best_model_info.id, best_model_info.loc, best_model_info.loc]) is None:
            raise RuntimeError(
                f"can not find the best model for dataset [{model_finder.dataset}], label [{model_finder.label}] "
                f"from wandb"
            )

        # we have a zip file, unzip it and return the model file from that dir.
        unzip_dir = os.path.expanduser(f"~/.cache/tmp/{best_model_info.id}")
        shutil.unpack_archive(filename=best_model_info.loc, extract_dir=unzip_dir, format="zip")
        try:
            model_loc = [os.path.join(unzip_dir, x) for x in os.listdir(unzip_dir) if x.endswith(".pyt")][0]
        except IndexError:
            raise RuntimeError(f"can not unzip model file for wandb run id {best_model_info.id}")
        print(
            f"the best model for dataset [{model_finder.dataset}], label [{model_finder.label}] from wandb is "
            f"{best_model_info.id}, and is stored at {model_loc}"
        )
        return model_loc


def main():
    parser = argparse.ArgumentParser("test")
    parser.add_argument("--config_file", default="config/ewt-pos.yml")
    args = parser.parse_args()
    exp_params = read_json_or_yml(args.config_file)
    _multi_label = False
    if exp_params["loader"]["multi_label"]:
        _multi_label = True
    print("loading data...")
    data = get_data(exp_params["loader"], exp_params["dataset"]["path"])
    print("data loaded, loading model..")
    model = torch.load(get_model(exp_params["model_loc_or_config"]), map_location=get_device())
    if _multi_label:
        trainer = create_trainer(model, trainer_type="mlc")
    else:
        trainer = create_trainer(model)
    print("model loaded, loading reporters..")
    for filt, dt in data.test.items():  # run for every filter on test data, including none
        reporters = create_test_reporters(exp_params["reporting"], config_file=args.config_file, filt=filt)
        reporting_fns = [x.step for x in reporters]
        print("reporters loaded, testing..")

        test_metrics = trainer.test(dt, reporting_fns=reporting_fns, phase="Test")
        print(test_metrics)
        for reporter in reporters:
            reporter.done(checkpoint_needed=False)


if __name__ == "__main__":
    main()

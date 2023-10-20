"""
given a project and a label, search the best model from the project with that label.
You need to have wandb installed on your env, and logged in:
```
pip install wandb
wandb login
```
"""

import wandb
from typing import Dict, Optional
from dataclasses import dataclass
import os
import shutil
import json


WANDB_CACHE_DIR = os.path.expanduser("~/.wandb-runs-cache")
WANDB_CACHE_FILE = "cache.json"


@dataclass
class BestModelInfo:
    id: Optional[str]
    loc: Optional[str]
    config: Optional[Dict]


class BestModelFinder:
    """
    Find the best ML model for a dataset and label
    """

    def __init__(self, dataset: str, label: str):
        self.dataset = dataset
        self.label = label

    def run(self, metric: str) -> BestModelInfo:
        """
        filter the results by a function
        :param metric:
        :return:
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


def get_from_cache(best_model, best_model_name, _id, cache_dir=WANDB_CACHE_DIR, cache_file=WANDB_CACHE_FILE):
    cache_file = os.path.join(cache_dir, cache_file)
    existing_files = json.load(open(cache_file)) if os.path.exists(cache_file) else {}
    if _id in existing_files:
        print(f"files for run {_id} found in cache")
        return existing_files[_id]
    print(f"files for run {_id} not found in cache, downloading..")
    os.makedirs(cache_dir, exist_ok=True)
    best_model.download(root=cache_dir, replace=True)
    download_loc = os.path.join(cache_dir, f"{best_model_name}")
    new_loc = os.path.join(cache_dir, _id)
    shutil.copy(download_loc, new_loc)
    os.remove(download_loc)
    print("downloaded")
    existing_files.update({_id: new_loc})
    json.dump(existing_files, open(cache_file, "w"), indent=2)
    return new_loc


class BestModelFinderWandb(BestModelFinder):
    def __init__(self, dataset: str, label: str):
        super().__init__(dataset=dataset, label=label)
        api = wandb.Api()
        _id_str = f"eptests/{dataset}"
        self.runs = [x for x in api.runs(_id_str) if label == x.config["label"]]
        if not self.runs:
            raise RuntimeError("no result for this experiment in wandb")
        print(f"{len(self.runs)} runs found")

    def run(self, metric: str, *args, **kwargs) -> BestModelInfo:
        """
        download the best file if possible and return the location on disk, else return None
        :param metric:
        :return:
        """
        best_model = None
        best_model_name = None
        best_config = None
        best_run_id = None
        min_is_better = False
        if metric[0] == "-":
            min_is_better = True
            metric = metric[1:]
        best_value = 0
        multiplier = 1
        if min_is_better:
            best_value = 1e10
            multiplier = -1
        for run in self.runs:
            summary = run.summary
            if not summary.keys():  # bad runs, TODO: filter properly
                continue
            if summary["phase"] == "Test":
                current_value = summary[metric]
                if current_value * multiplier > best_value * multiplier:
                    best_value = current_value
                    best_model_name = [x.name for x in run.files() if x.name.endswith(".zip")][0]
                    best_model = run.file(best_model_name)
                    if best_model.md5 == "0":  # empty
                        best_model = None
                    if best_model is not None:
                        best_config = run.config
                        best_run_id = run.id
        if best_model is not None:
            assert best_model_name is not None
            loc = get_from_cache(best_model, best_model_name, best_run_id)
            return BestModelInfo(id=best_run_id, loc=loc, config=best_config)
        print(best_model)
        return BestModelInfo(None, None, None)


if __name__ == "__main__":
    b = BestModelFinderWandb(dataset="ewt-pos", label="default_label")
    result = b.run(metric="acc")
    print(result)

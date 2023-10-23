import datetime
import torch
from typing import Dict, List, Mapping, Tuple
from tqdm import tqdm
import json
import yaml
from eptests.libs.yamlenv import load as yaml_load


YML_EXTENSIONS = ["yml", "yaml"]


def good_update_interval(total_iters, num_desired_updates):
    """
    This function will try to pick an intelligent progress update interval
    based on the magnitude of the total iterations.

    Parameters:
      `total_iters` - The number of iterations in the for-loop.
      `num_desired_updates` - How many times we want to see an update over the
                              course of the for-loop.
    """

    exact_interval = total_iters / num_desired_updates
    order_of_mag = len(str(total_iters)) - 1

    # Our update interval should be rounded to an order of magnitude smaller.
    round_mag = order_of_mag - 1

    # Round down and cast to an int.
    update_interval = int(round(exact_interval, -round_mag))

    # Don't allow the interval to be zero!
    if update_interval == 0:
        update_interval = 1
    return update_interval


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def remove_key_from_dict(_dict: Dict, _key: str = "name") -> Dict:
    """
    remove a key from a dictionary and return a new one
    :param _dict:
    :param _key:
    :return:
    """
    return {k: v for k, v in _dict.items() if k != _key}


def read_json_or_yml(path):
    if path.endswith("jsonl"):
        con = []
        for line in tqdm(open(path)):
            con.append(json.loads(line))
        return con
    if path.endswith("json"):
        return json.load(open(path))
    elif any([path.endswith(ext) for ext in YML_EXTENSIONS]):
        return yaml_load(open(path))
    raise RuntimeError("config must be json or yaml")


def write_file(data, path, mode="w", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        f.write(data)


def rec_val_replace(obj, replacements: List[Tuple[str, str]]):
    if isinstance(obj, Mapping):
        return {key: rec_val_replace(val, replacements)
                for key, val in obj.items()}
    elif isinstance(obj, str):
        for to_replace, replace_with in replacements:
            obj = obj.replace(to_replace, replace_with)
        return obj
    return obj


def write_json_yml(data, path, yml=False, replacements: List[Tuple[str, str]] = []):
    if not yml:
        return write_file(json.dumps(data, indent=2), path)
    if not replacements:
        yaml.safe_dump(data, open(path, 'w'), default_flow_style=False)
        return
    data = rec_val_replace(data, replacements)
    yaml.safe_dump(data, open(path, 'w'),  default_flow_style=False, sort_keys=False)
    return

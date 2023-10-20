import sys
import os
import json
from tqdm import tqdm


def write_jsonl(_phase_data, _file):
    with open(_file, "w") as wf:
        for item in tqdm(_phase_data, desc=f"writing {phase}"):
            wf.write(f"{json.dumps(item)}\n")


json_file = sys.argv[1]
print("reading data")
data = json.load(open(json_file))
base_dir = os.path.abspath(json_file[:-5])
os.makedirs(base_dir, exist_ok=True)
phases = ["train", "dev", "test"]
for phase in phases:
    write_jsonl(data[phase], f"{base_dir}/{phase}.jsonl")
json.dump(data["metadata"], open(f"{base_dir}/metadata.json", "w"), indent=2)
json.dump(data["labels"], open(f"{base_dir}/labels.json", "w"), indent=2)

from __future__ import print_function

import getpass
import socket
import os
import shutil
from baseline.reporting import EpochReportingHook
from mead.utils import read_config_file_or_json
from baseline.reporting import register_reporting
import wandb
from mead.utils import hash_config


@register_reporting(name="wandb")
class WandbReporting(EpochReportingHook):
    def __init__(self, **kwargs):
        super(WandbReporting, self).__init__(**kwargs)
        self.exp_config = read_config_file_or_json(kwargs["config_file"])
        self.task = kwargs["task"]
        self.dataset = self.exp_config["dataset"]["name"]
        self.label = self.exp_config["label"]
        # label = kwargs.get("label", "default_label")
        username = kwargs.get("user", getpass.getuser())
        hostname = kwargs.get("host", socket.gethostname())
        if "filt" in kwargs:
            self.exp_config.update({"filt": kwargs.get("filt", "")})
        self.exp_config.update({"username": username, "hostname": hostname})
        self.exp_config.update({"config_hash": hash_config(self.exp_config)})
        wandb.init(project=self.dataset, config=self.exp_config, entity="eptests")

    def _step(self, metrics, tick, phase, tick_type, **kwargs):
        """Write intermediate results to a logging memory object that ll be pushed to wandb

        :param metrics: A map of metrics to scores
        :param tick: The time (resolution defined by `tick_type`)
        :param phase: The phase of training (`Train`, `Valid`, `Test`)
        :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
        :return:
        """
        msg = {"tick_type": tick_type, "tick": tick, "phase": phase}
        for k, v in metrics.items():
            msg[k] = v
        wandb.log(msg)

    def done(self, checkpoint_needed=True):
        """Write the log to the wandb database"""
        if checkpoint_needed:
            checkpoint_base = self._search_checkpoint_base()
            if checkpoint_base is None:
                wandb.finish(exit_code=1)
                raise RuntimeError("checkpoint is none")
            print("pushing to wandb server")
            wandb.save(checkpoint_base)
            wandb.finish(exit_code=0)
        else:  # checkpoint is not needed when using just test data
            wandb.finish(exit_code=0)

    def _search_checkpoint_base(self):
        """Finds if the checkpoint exists as a zip file or a bunch of files."""
        model_file = f"classify-model-{os.getpid()}.pyt"
        label_file = f"classify-model-{os.getpid()}.labels"
        if os.path.exists(model_file) and os.path.exists(label_file):
            _dir = f"classify-{os.getpid()}"
            os.makedirs(_dir)
            shutil.move(model_file, _dir)
            shutil.move(label_file, _dir)
            print("zipping model files")
            os.remove(f"{self.label}-{os.getpid()}.zip") if os.path.exists(f"{self.label}-{os.getpid()}.zip") else None
            shutil.make_archive(f"{self.label}-{os.getpid()}", "zip", _dir)
            return f"{self.label}-{os.getpid()}.zip"
        return None


def create_reporting_hook(**kwargs):
    return WandbReporting(**kwargs)

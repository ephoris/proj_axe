#!/usr/bin/env python
import logging
import os
import sys
import toml
from typing import Any

from jobs.create_lcm_data import CreateLCMData
from jobs.train_lcm import TrainLCM
from jobs.create_ltuner_data import CreateLTunerData
from jobs.train_ltuner import TrainLTuner


class AxeDriver:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        logging.basicConfig(**config["log"])
        self.log: logging.Logger = logging.getLogger(config["app"]["name"])
        self.log.info(
            f"Log level: {logging.getLevelName(self.log.getEffectiveLevel())}"
        )

    def run(self):
        self.log.info(f'Staring app {self.config["app"]["name"]}')

        jobs = {
            "create_lcm_data": CreateLCMData,
            "train_lcm": TrainLCM,
            "create_ltuner_data": CreateLTunerData,
            "train_ltuner": TrainLTuner,
        }
        jobs_list = self.config["app"]["run"]
        for job_name in jobs_list:
            job = jobs.get(job_name, None)
            if job is None:
                self.log.warning(f"No job associated with {job_name}")
                continue
            job = job(config)
            _ = job.run()

        self.log.info("All jobs finished, exiting")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        file_dir = os.path.dirname(__file__)
        config_path = os.path.join(file_dir, "axe.toml")

    with open(config_path) as fid:
        config = toml.load(fid)

    driver = AxeDriver(config)
    driver.run()

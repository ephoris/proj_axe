#!/usr/bin/env python
import argparse
import logging
import multiprocessing as mp
import os
import toml

import pyarrow as pa
import pyarrow.parquet as pq
from axe.lsm.types import LSMBounds, Policy
from axe.lcm.data.schema import LCMDataSchema

from tqdm import tqdm


class CreateLCMData:
    def __init__(self, cfg: dict) -> None:
        self.log = logging.getLogger(cfg["log"]["name"])
        self.disable_tqdm: bool = cfg["log"]["disable_tqdm"]
        self.policy: Policy = getattr(Policy, cfg["lsm"]["policy"])
        self.bounds: LSMBounds = LSMBounds(**cfg["lsm"]["bounds"])
        self.seed: int = cfg["app"]["random_seed"]

        jcfg = cfg["job"]["create_lcm_data"]
        self.output_dir: str = jcfg["output_dir"]
        self.num_samples: int = jcfg["num_samples"]
        self.num_files: int = jcfg["num_files"]
        self.file_prefix: str = jcfg["file_prefix"]
        self.num_workers: int = jcfg["num_workers"]
        self.overwrite_if_exists: bool = jcfg["overwrite_if_exists"]

    def generate_parquet_file(self, schema: LCMDataSchema, idx: int, pos: int) -> int:
        fname = f"{self.file_prefix}_{idx:04}.parquet"
        fpath = os.path.join(self.output_dir, fname)

        if os.path.exists(fpath) and (not self.overwrite_if_exists):
            self.log.info(f"{fpath} exists, exiting.")
            return -1

        pbar = tqdm(
            range(self.num_samples),
            desc=fname,
            position=pos,
            ncols=80,
            disable=self.disable_tqdm,
        )
        table = [schema.sample_row_dict() for _ in pbar]
        table = pa.Table.from_pylist(table)
        pq.write_table(table, fpath)

        return idx

    def generate_file(self, idx: int, single_worker: bool = False) -> int:
        pos = 0
        if len(mp.current_process()._identity) > 0 and not single_worker:
            pos = mp.current_process()._identity[0] - 1
        schema = LCMDataSchema(self.policy, self.bounds, seed=(self.seed + idx))

        self.generate_parquet_file(schema, idx, pos)

        return idx

    def run(self) -> None:
        self.log.info("Running data generator for LCM")
        os.makedirs(self.output_dir, exist_ok=True)
        self.log.info(f"Writing all files to {self.output_dir}")

        inputs = list(range(0, self.num_files))
        threads = self.num_workers
        if threads == -1:
            threads = mp.cpu_count()
        if threads > self.num_files:
            self.log.info("Num workers > num files, scaling down")
            threads = self.num_files
        self.log.debug(f"Using {threads=}")

        if threads == 1:
            for idx in range(self.num_files):
                self.generate_file(idx, single_worker=True)
        else:
            with mp.Pool(
                threads, initializer=tqdm.set_lock, initargs=(mp.RLock(),)
            ) as p:
                p.map(self.generate_file, inputs)

        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file")
    args = parser.parse_args()
    job = CreateLCMData(toml.load(args.config))

    job.run()


if __name__ == "__main__":
    main()

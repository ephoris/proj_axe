#!/usr/bin/env python
import argparse
import logging
import os
import csv
from typing import Optional, Tuple
from torch import Tensor
from tqdm import tqdm

import toml
import torch
import polars as pl
from axe.lcm.data.dataset import CostModelDataSet
from axe.lcm.data.schema import LCMDataSchema
from axe.lcm.model.builder import LearnedCostModelBuilder
from axe.lsm.types import LSMBounds, Policy
from axe.util.losses import LossBuilder
from axe.util.lr_scheduler import LRSchedulerBuilder
from axe.util.optimizer import OptimizerBuilder
from torch.utils.data import DataLoader


class TrainLCM:
    def __init__(self, cfg: dict) -> None:
        self.log: logging.Logger = logging.getLogger(cfg["app"]["name"])
        self.disable_tqdm: bool = cfg["app"]["disable_tqdm"]
        self.use_gpu = cfg["job"]["use_gpu_if_avail"]
        self.device = (
            torch.device("cuda")
            if self.use_gpu and torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.policy: Policy = getattr(Policy, cfg["lsm"]["policy"])
        self.bounds: LSMBounds = LSMBounds(**cfg["lsm"]["bounds"])
        self.schema = LCMDataSchema(self.policy, self.bounds)
        self.jcfg = cfg["job"]["train_lcm"]
        self.cfg = cfg

        # Build everything we need for training
        self.model = self._build_model()
        self.loss_fn = self._build_loss_fn()
        self.optimizer = self._build_optimizer(self.model)
        self.scheduler = self._build_scheduler(self.optimizer)
        self.train_data, self.validate_data = self._build_data()

    def _build_loss_fn(self) -> torch.nn.Module:
        choice = self.jcfg["loss_fn"]
        loss = LossBuilder(self.cfg["loss"]).build(choice)
        self.log.info(f"Loss function: {choice}")
        if loss is None:
            self.log.warning(f"Invalid loss function: {choice}")
            raise KeyError
        if self.use_gpu and torch.cuda.is_available():
            loss.to("cuda")

        return loss

    def _build_model(self) -> torch.nn.Module:
        model = LearnedCostModelBuilder(
            self.bounds, **self.cfg["lcm"]["model"]
        ).build_model(self.policy)
        if self.use_gpu and torch.cuda.is_available():
            model.to("cuda")

        return model

    def _build_optimizer(self, model) -> torch.optim.Optimizer:
        return OptimizerBuilder(self.cfg["optimizer"]).build_optimizer(
            optimizer_choice=self.jcfg["optimizer"], model=model
        )

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        return LRSchedulerBuilder(self.cfg["scheduler"]).build_scheduler(
            optimizer, self.jcfg["lr_scheduler"]
        )

    def _build_data(self) -> Tuple[DataLoader, DataLoader]:
        self.log.info(f"Data directory: {self.jcfg['data_dir']}")
        table = pl.read_parquet(self.jcfg["data_dir"])
        table = table.with_columns(pl.all().shuffle(seed=1)).with_row_index()
        training_table = table.filter(
            pl.col("index") < pl.col("index").max() * self.jcfg["data_split"]
        )
        training_dataset = CostModelDataSet(
            table=training_table,
            bounds=self.bounds,
            policy=self.policy,
            one_hot_transform=True,
        )
        training_data = DataLoader(
            dataset=training_dataset,
            batch_size=self.jcfg["batch_size"],
            num_workers=self.jcfg["num_workers"],
            shuffle=True,
        )
        validate_table = table.filter(
            pl.col("index") >= pl.col("index").max() * self.jcfg["data_split"]
        )
        validate_dataset = CostModelDataSet(
            table=validate_table,
            bounds=self.bounds,
            policy=self.policy,
            one_hot_transform=True,
        )
        validate_data = DataLoader(
            dataset=validate_dataset,
            batch_size=self.jcfg["batch_size"],
            num_workers=self.jcfg["num_workers"],
        )

        return training_data, validate_data

    def _make_save_dir(self) -> None:
        self.log.info(f"Saving tuner in {self.jcfg['save_dir']}")
        try:
            os.makedirs(self.jcfg["save_dir"], exist_ok=False)
        except FileExistsError:
            return None
        with open(os.path.join(self.jcfg["save_dir"], "axe.toml"), "w") as fid:
            toml.dump(self.cfg, fid)
        os.makedirs(os.path.join(self.jcfg["save_dir"], "checkpoints"))

    def train_step(self, feats: Tensor, labels: Tensor, **kwargs) -> float:
        label = labels.to(self.device)
        feats = feats.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(feats, **kwargs)
        loss = self.loss_fn(pred, label)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_loop(self) -> float:
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_data, ncols=80, disable=self.disable_tqdm)
        for batch, (feats, labels) in enumerate(pbar):
            loss = self.train_step(feats, labels)
            if batch % (25) == 0:
                pbar.set_description(f"loss {loss:e}")
            total_loss += loss
            if self.scheduler is not None:
                self.scheduler.step()

        return total_loss / len(self.train_data)

    def validate_step(self, feats: Tensor, labels: Tensor, **kwargs) -> float:
        with torch.no_grad():
            labels = labels.to(self.device)
            feats = feats.to(self.device)
            pred = self.model(feats, **kwargs)
            validate_loss = self.loss_fn(pred, labels).item()

        return validate_loss

    def validate_loop(self) -> float:
        self.model.eval()
        test_loss = 0
        pbar = tqdm(self.validate_data, ncols=80, disable=self.disable_tqdm)
        for feats, labels in pbar:
            loss = self.validate_step(feats, labels)
            pbar.set_description(f"validate {loss:e}")
            test_loss += loss

        return test_loss

    def save_model(self, checkpoint_name: str, **kwargs) -> None:
        checkpoint_dir = os.path.join(self.jcfg["save_dir"], "checkpoints")
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        save_dict.update(kwargs)
        torch.save(save_dict, os.path.join(checkpoint_dir, checkpoint_name))

        return

    def run(self):
        self._make_save_dir()

        loss_file = os.path.join(self.jcfg["save_dir"], "losses.csv")
        with open(loss_file, "w") as fid:
            loss_csv_write = csv.writer(fid)
            loss_csv_write.writerow(["epoch", "train_loss", "test_loss"])

        max_epochs = self.jcfg["max_epochs"]
        loss_min = self.validate_loop()
        for epoch in range(max_epochs):
            self.log.info(f"Epoch: [{epoch+1}/{max_epochs}]")
            train_loss = self.train_loop()
            curr_loss = self.validate_loop()
            self.log.info(f"Training loss: {train_loss}")
            self.log.info(f"Validate loss: {curr_loss}")
            if not self.jcfg["no_checkpoint"]:
                self.save_model(f"epoch{epoch:02d}.model", loss=curr_loss)

            if curr_loss < loss_min:
                loss_min = curr_loss
                self.log.info("New minmum loss, saving...")
                self.save_model("best.model", loss=loss_min, epoch=epoch)
            with open(loss_file, "a") as fid:
                write = csv.writer(fid)
                write.writerow([epoch + 1, train_loss, curr_loss])

        self.log.info("Training finished")

        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file")
    args = parser.parse_args()
    config = toml.load(args.config)
    logging.basicConfig(**config["log"])
    log: logging.Logger = logging.getLogger(config["app"]["name"])
    log_level = logging.getLevelName(log.getEffectiveLevel())
    log.debug(f"Log level: {log_level}")
    job = TrainLCM(config)

    job.run()


if __name__ == "__main__":
    main()

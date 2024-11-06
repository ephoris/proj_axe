from typing import Callable, Optional, Tuple

import torch
import polars as pl

from axe.lsm.types import LSMBounds, Policy


class CostModelDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        feat_cols: list[str],
        label_cols: list[str],
        bounds: LSMBounds,
        policy: Policy,
        feat_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
    ) -> None:
        self.bounds: LSMBounds = bounds
        self.policy = policy
        self.data_path: str = data_path
        self.feat_cols = feat_cols
        self.label_cols = label_cols
        self.feat_transform: Optional[Callable] = feat_transform
        self.label_transform: Optional[Callable] = label_transform

        table = pl.read_parquet(data_path)
        table = self.preprocess_func(table)
        self.table: pl.DataFrame = table

    def preprocess_func(self, table: pl.DataFrame):
        table = table.with_columns(pl.col("T").sub(self.bounds.size_ratio_range[0]))
        if self.policy == Policy.QHybrid:
            table = table.with_columns(pl.col("Q").sub(self.bounds.size_ratio_range[0]))
        elif self.policy == Policy.Fluid:
            table = table.with_columns(
                pl.col("Y").sub(self.bounds.size_ratio_range[0]),
                pl.col("Z").sub(self.bounds.size_ratio_range[0]),
            )
        elif self.policy == Policy.Kapacity:
            table = table.with_columns(
                [
                    pl.col(f"K_{i}").sub(self.bounds.size_ratio_range[0]).clip(0)
                    for i in range(self.bounds.max_considered_levels)
                ]
            )

        return table

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = torch.Tensor(self.table[idx, self.feat_cols].to_numpy())
        labels = torch.Tensor(self.table[idx, self.label_cols].to_numpy())

        return feats, labels

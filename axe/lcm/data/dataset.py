from typing import Tuple

import torch.nn.functional as F
import torch
import polars as pl

from axe.lcm.data.schema import LCMDataSchema
from axe.lsm.types import LSMBounds, Policy


class CostModelDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        table: pl.DataFrame,
        bounds: LSMBounds,
        policy: Policy,
        one_hot_transform: bool = False,
        preprocess: bool = True,
    ) -> None:
        self.bounds: LSMBounds = bounds
        self.policy: Policy = policy
        self.one_hot_transform: bool = one_hot_transform
        self.schema: LCMDataSchema = LCMDataSchema(policy, bounds)
        self.categories = bounds.size_ratio_range[1] - bounds.size_ratio_range[0]

        if preprocess:
            table = self.preprocess_func(table)
        self.table: pl.DataFrame = table

    def preprocess_func(self, table: pl.DataFrame):
        table = table.with_columns(
            pl.col("size_ratio").sub(self.bounds.size_ratio_range[0])
        )
        if self.policy == Policy.QHybrid:
            table = table.with_columns(
                pl.col("Q").sub(self.bounds.size_ratio_range[0] - 1)
            )
        elif self.policy == Policy.Fluid:
            table = table.with_columns(
                pl.col("Y").sub(self.bounds.size_ratio_range[0] - 1),
                pl.col("Z").sub(self.bounds.size_ratio_range[0] - 1),
            )
        elif self.policy == Policy.Kapacity:
            table = table.with_columns(
                [
                    pl.col(f"K_{i}").sub(self.bounds.size_ratio_range[0] - 1).clip(0)
                    for i in range(self.bounds.max_considered_levels)
                ]
            )

        return table

    def get_feats(self, idx: int) -> torch.Tensor:
        if self.one_hot_transform:
            row = self.table[idx, self.schema.feat_cols()].to_torch(return_type="dict")
            for col_name in self.schema.categorical_feats():
                row[col_name] = F.one_hot(row[col_name], self.categories).flatten()
            out = torch.cat(list(row[col_name] for col_name in self.schema.feat_cols()))
        else:
            out = self.table[idx, self.schema.feat_cols()].to_torch()

        return out

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.get_feats(idx)
        feats = feats.to(torch.float)
        labels = self.table[idx, self.schema.label_cols()].to_torch().flatten()
        labels = labels.to(torch.float)

        return feats, labels

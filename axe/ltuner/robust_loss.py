from typing import Any
import os

import torch
import toml
from torch import Tensor

from axe.lcm.data.schema import LCMDataSchema
from axe.lcm.model.builder import LearnedCostModelBuilder
from axe.lsm.types import Policy, LSMBounds
from axe.ltuner.data.schema import LTunerDataSchema


class LearnedRobustLoss(torch.nn.Module):
    def __init__(self, config: dict[str, Any], model_path: str):
        self.penalty_factor = config["ltune"]["penalty_factor"]
        self.bounds: LSMBounds = LSMBounds(**config["lsm"]["bounds"])
        self.policy: Policy = getattr(Policy, config["lsm"]["policy"])
        min_t, max_t = self.bounds.size_ratio_range
        self.kap_categories = max_t - min_t
        self.tuner_schema = LTunerDataSchema(
            policy=self.lcm_policy,
            bounds=self.lcm_bounds,
            robust=True,
        )

        lcm_cfg = toml.load(os.path.join(model_path, "axe.toml"))
        self.lcm_bounds: LSMBounds = LSMBounds(**lcm_cfg["lsm"]["bounds"])
        self.lcm_policy: Policy = getattr(Policy, lcm_cfg["lsm"]["policy"])
        self.lcm_schema = LCMDataSchema(policy=self.lcm_policy, bounds=self.lcm_bounds)
        self.mem_budget_idx = self.lcm_schema.feat_cols().index("mem_budget")

        # Before building we assert LCM -> LTuner designs are matching
        assert self.bounds == self.lcm_bounds
        assert self.policy == self.lcm_policy

        self.lcm = LearnedCostModelBuilder(
            schema=self.lcm_schema,
            **lcm_cfg["lcm"]["model"],
        ).build(disable_one_hot_encoding=True)

        data = torch.load(os.path.join(model_path, "best_model.model"))
        status = self.model.load_state_dict(data["model_state_dict"])
        assert len(status.missing_keys) == 0
        assert len(status.unexpected_keys) == 0
        self.model.eval()

    def kl_div_conj(self, input):
        ret = torch.exp(input) - 1

        return ret

    def calc_mem_penalty(self, label, bpe):
        mem_budget = label[:, self.mem_budget_idx].view(-1, 1)
        penalty = torch.zeros(bpe.size()).to(bpe.device)
        idx = bpe >= mem_budget
        penalty[idx] = self.penalty_factor * (bpe[idx] - mem_budget[idx])
        idx = bpe < 0
        penalty[idx] = self.penalty_factor * (0 - bpe[idx])

        bpe[bpe > mem_budget] = mem_budget[bpe > mem_budget]
        bpe[bpe < 0] = 0

        return bpe, penalty

    def split_tuner_out(self, tuner_out):
        # First 3 items are going to be lagragians then BPE.
        # After we have categorical features (size ratio and capacities)
        # [lagrag0, lagrag1, bpe, size_ratio, k1, ..., kL]
        eta = tuner_out[:, 0].view(-1, 1)
        lamb = tuner_out[:, 1].view(-1, 1)
        bpe = tuner_out[:, 2]
        bpe = bpe.view(-1, 1)
        categorical_feats = tuner_out[:, 3:]

        return eta, lamb, bpe, categorical_feats

    def _forward_impl(self, pred, label):
        # For learned cost model loss, the prediction is the DB configuration
        # and label is the workload and system params, rho at the end
        eta, lamb, bpe, categorical_feats = self.split_tuner_out(pred)
        rho = label[:, self.tuner_schema.feat_cols().index("rho")]
        sys_label = label[:, :self.tuner_schema.feat_cols().index("rho")]
        bpe, penalty = self.calc_mem_penalty(sys_label, bpe)

        inputs = torch.concat([sys_label, bpe, categorical_feats], dim=-1)
        # print(f"{label.shape=}")
        # print(f"{rho.shape=}")
        # print(f"{sys_label.shape=}")
        # print(f"{inputs.shape=}")
        out = self.model(inputs)
        # print(f"{eta.shape=}")
        # print(f"{lamb.shape=}")
        # print(f"{out.shape=}")
        out = (out - eta) / lamb
        out = self.kl_div_conj(out)
        # print(f"after kl_div_conj {out=}")
        out = out.sum(dim=-1)
        out = eta + (lamb * rho) + (lamb * out)
        out = out.square()
        out = out + penalty
        out = out.mean()

        return out

    def forward(self, pred: Tensor, label: Tensor):
        assert self.model.training is False
        out = self._forward_impl(pred, label)

        return out

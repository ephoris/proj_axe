from torch import nn
import torch

from axe.lcm.data.input_features import kINPUT_FEATS_DICT
from axe.lcm.model import KapLCM, QHybridLCM, ClassicLCM, FluidLCM
from axe.lsm.types import LSMBounds, Policy


class LearnedCostModelBuilder:
    def __init__(
        self,
        bounds: LSMBounds,
        hidden_length: int = 1,
        hidden_width: int = 64,
        embedding_size: int = 8,
        norm_layer: str = "Batch",
        policy_embedding_size: int = 2,  # only used for classic model
        decision_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        self.embedding_size = embedding_size
        self.policy_embedding_size = policy_embedding_size
        self.hidden_length = hidden_length
        self.hidden_width = hidden_width
        self.decision_dim = decision_dim
        self.dropout = dropout
        self.max_levels = bounds.max_considered_levels
        self.size_ratio_min, self.size_ratio_max = bounds.size_ratio_range
        self.capacity_range = self.size_ratio_max - self.size_ratio_min
        self.bounds = bounds

        self.norm_layer = nn.BatchNorm1d
        if norm_layer == "Layer":
            self.norm_layer = nn.LayerNorm

        self._models = {
            Policy.Kapacity: KapLCM,
            Policy.QHybrid: QHybridLCM,
            Policy.Classic: ClassicLCM,
            Policy.Fluid: FluidLCM,
        }

    def get_choices(self):
        return self._models.keys()

    def build_model(self, policy: Policy) -> torch.nn.Module:
        feats_list = kINPUT_FEATS_DICT.get(policy, None)
        if feats_list is None:
            raise TypeError("Illegal policy")

        num_feats = len(feats_list)
        if "K" in feats_list:
            # Add number of features to expand K to K0, K1, ..., K_maxlevels
            num_feats += self.max_levels - 1

        args = {
            "num_feats": num_feats,
            "capacity_range": self.capacity_range,
            "embedding_size": self.embedding_size,
            "hidden_length": self.hidden_length,
            "hidden_width": self.hidden_width,
            "dropout_percentage": self.dropout,
            "decision_dim": self.decision_dim,
            "norm_layer": self.norm_layer,
        }

        model_class = self._models.get(policy, None)
        if model_class is None:
            raise NotImplementedError("Model for policy not implemented")

        if model_class is ClassicLCM:
            args["policy_embedding_size"] = self.policy_embedding_size

        if model_class is KapLCM:
            args["max_levels"] = self.max_levels

        model = model_class(**args)

        return model

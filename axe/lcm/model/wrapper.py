import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from axe.lcm.data.schema import LCMDataSchema
from axe.lsm.types import LSMDesign, Policy, System, Workload


class LCMWrapper:
    def __init__(self, model: torch.nn.Module, schema: LCMDataSchema) -> None:
        model.eval()
        self.model = model
        self.schema = schema
        self.bounds = schema.bounds
        self.policy = schema.policy
        # TODO: Assert schema and models are matching
        # matching_policy_to_model = {
        #     Policy.Classic: ClassicLCM,
        #     Policy.Kapacity: KapLCM
        # }

    def convert_to_tensor(
        self,
        design: LSMDesign,
        system: System,
        workload: Workload,
    ) -> Tensor:
        min_t, max_t = self.bounds.size_ratio_range
        categories = max_t - min_t
        wl = [workload.z0, workload.z1, workload.q, workload.w]
        sys = [
            system.entries_per_page,
            system.selectivity,
            system.entry_size,
            system.mem_budget,
            system.num_entries,
        ]
        base_input = Tensor(wl + sys + [design.bits_per_elem])
        size_ratio = Tensor([np.ceil(design.size_ratio)]).to(torch.long) - min_t
        size_ratio = F.one_hot(size_ratio, num_classes=categories).flatten()
        if design.policy in (Policy.Tiering, Policy.Leveling, Policy.Classic):
            policy = Tensor([design.policy.value]).to(torch.long)
            policy = F.one_hot(policy, num_classes=2).flatten()
            out = torch.concat((base_input, size_ratio, policy))
        elif design.policy == Policy.Kapacity:
            ks = Tensor(design.kapacity).to(torch.long) - 1
            ks = F.pad(
                ks.clamp(min=0),
                (0, self.bounds.max_considered_levels - ks.shape[0]),
                "constant",
                0,
            )
            ks = F.one_hot(ks, num_classes=categories).flatten()
            out = torch.concat((base_input, size_ratio, ks))
        else:  # (design.policy == Policy.Fluid) or (design.policy == Policy.QHybrid)
            kaps = Tensor(design.kapacity).to(torch.long) - 1
            kaps = F.one_hot(kaps, num_classes=categories).flatten()
            out = torch.concat((base_input, size_ratio, kaps))

        return out

    def __call__(
        self,
        design: LSMDesign,
        system: System,
        workload: Workload,
    ) -> float:
        x = self.convert_to_tensor(design=design, system=system, workload=workload)
        x = x.to(torch.float).view(1, -1)
        with torch.no_grad():
            pred = self.model(x)
            pred = pred.sum().item()

        return pred

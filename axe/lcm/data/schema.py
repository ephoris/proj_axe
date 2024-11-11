import axe.lsm.data_generator as DataGen
from axe.lsm.cost import Cost
from axe.lsm.types import LSMBounds, LSMDesign, Policy, System, Workload

kSYSTEM_HEADER = ["entry_p_page", "selec", "entry_size", "mem_budget", "num_elem"]
kWORKLOAD_HEADER = ["z0", "z1", "q", "w"]
kCOST_HEADER = ["z0_cost", "z1_cost", "q_cost", "w_cost"]


class LCMDataSchema:
    def __init__(
        self, policy: Policy, bounds: LSMBounds, precision: int = 3, seed: int = 0
    ) -> None:
        self.cost: Cost = Cost(bounds.max_considered_levels)
        self.gen = DataGen.build_data_gen(policy=policy, bounds=bounds, seed=seed)
        self.policy: Policy = policy
        self.bounds: LSMBounds = bounds
        self.precision: int = precision

    def label_cols(self) -> list[str]:
        return kCOST_HEADER

    def feat_cols(self) -> list[str]:
        base_feat_cols = kWORKLOAD_HEADER + kSYSTEM_HEADER
        if (self.policy == Policy.Tiering) or (self.policy == Policy.Leveling):
            return base_feat_cols + ["bits_per_elem", "size_ratio"]
        elif self.policy == Policy.Classic:
            return base_feat_cols + ["bits_per_elem", "size_ratio", "policy"]
        elif self.policy == Policy.QHybrid:
            return base_feat_cols + ["bits_per_elem", "size_ratio", "Q"]
        elif self.policy == Policy.Fluid:
            return base_feat_cols + ["bits_per_elem", "size_ratio", "Y", "Z"]
        elif self.policy == Policy.Kapacity:
            cols = [f"K_{i}" for i in range(self.bounds.max_considered_levels)]
            return base_feat_cols + ["bits_per_elem", "size_ratio"] + cols
        else:
            raise NotImplementedError

    def categorical_feats(self) -> list[str]:
        if (self.policy == Policy.Tiering) or (self.policy == Policy.Leveling):
            return ["size_ratio"]
        elif self.policy == Policy.Classic:
            return ["policy", "size_ratio"]
        elif self.policy == Policy.QHybrid:
            return ["size_ratio", "Q"]
        elif self.policy == Policy.Fluid:
            return ["size_ratio", "Y", "Z"]
        elif self.policy == Policy.Kapacity:
            cols = [f"K_{i}" for i in range(self.bounds.max_considered_levels)]
            return ["size_ratio"] + cols
        else:
            raise NotImplementedError

    def get_column_names(self) -> list[str]:
        column_names = (
            kCOST_HEADER
            + kWORKLOAD_HEADER
            + kSYSTEM_HEADER
            + ["bits_per_elem", "size_ratio", "policy"]
        )
        if self.policy == Policy.QHybrid:
            column_names += ["Q"]
        elif self.policy == Policy.Fluid:
            column_names += ["Y", "Z"]
        elif self.policy == Policy.Kapacity:
            column_names += [f"K_{i}" for i in range(self.bounds.max_considered_levels)]

        return column_names

    def sample_row(self) -> list:
        workload: Workload = self.gen.sample_workload()
        system: System = self.gen.sample_system()
        design: LSMDesign = self.gen.sample_design(system)

        line = [
            workload.z0 * self.cost.Z0(design, system),
            workload.z1 * self.cost.Z1(design, system),
            workload.q * self.cost.Q(design, system),
            workload.w * self.cost.W(design, system),
            workload.z0,
            workload.z1,
            workload.q,
            workload.w,
            system.entries_per_page,
            system.selectivity,
            system.entry_size,
            system.mem_budget,
            system.num_entries,
            design.bits_per_elem,
            design.size_ratio,
            design.policy.value,
        ] + list(design.kapacity)

        return line

    def sample_row_dict(self) -> dict:
        column_names = self.get_column_names()
        row = self.sample_row()
        line = {}
        for key, val in zip(column_names, row):
            line[key] = val

        return line

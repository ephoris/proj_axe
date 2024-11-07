import axe.lsm.data_generator as DataGen
from axe.lsm.cost import Cost
from axe.lsm.types import LSMBounds, LSMDesign, Policy, System, Workload

kSYSTEM_HEADER = ["entry_p_page", "selec", "entry_size", "mem_budget", "num_elem"]
kWORKLOAD_HEADER = ["z0", "z1", "q", "w"]
kCOST_HEADER = ["z0_cost", "z1_cost", "q_cost", "w_cost"]


class LCMDataSchema:
    def __init__(self, policy: Policy, bounds: LSMBounds, precision: int = 3) -> None:
        self.cost: Cost = Cost(bounds.max_considered_levels)
        self.gen = DataGen.build_data_gen(policy=policy, bounds=bounds)

        self.policy: Policy = policy
        self.bounds: LSMBounds = bounds
        self.precision: int = precision

    def get_column_names(self) -> list[str]:
        column_names = (
            kCOST_HEADER
            + kWORKLOAD_HEADER
            + kSYSTEM_HEADER
            + ["policy", "bits_per_elem", "size_ratio"]
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
            design.policy.value,
            design.bits_per_elem,
            design.size_ratio,
        ] + list(design.kapacity)

        return line

    def sample_row_dict(self) -> dict:
        column_names = self.get_column_names()
        row = self.sample_row()
        line = {}
        for key, val in zip(column_names, row):
            line[key] = val

        return line


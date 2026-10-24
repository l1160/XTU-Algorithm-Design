"""Unchanged repair/physics with initialization and velocity-reset telemetry."""
from repair_ablation_model import RepairFarm as BaseRepairFarm
import numpy as np
import algorithms as A
import variants as V


class RepairFarm(BaseRepairFarm):
    def __init__(self, *args, **kwargs):
        self.initial_keys = []
        super().__init__(*args, **kwargs)

    def evaluate(self, keys):
        result = super().evaluate(keys)
        if len(self.records) <= self.shared_initial_requests:
            self.initial_keys.append(result[0].copy())
        return result


def evaluate_with_velocity_diagnostics(farm, x, v=None):
    x = np.atleast_2d(x)
    start = len(farm.records)
    result = V.ORIGINAL_EVALUATE(farm, x, v)
    displaced = np.abs(result.x-x) > 1e-12
    incoming = np.zeros_like(x) if v is None else v
    for i in range(len(x)):
        farm.repair_details[start+i]["velocity_coordinates_reset"] = int(np.count_nonzero(displaced[i] & (incoming[i] != 0)))
    return result


def configure():
    V.configure({})
    A.evaluate = evaluate_with_velocity_diagnostics

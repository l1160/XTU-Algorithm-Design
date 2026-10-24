"""Telemetry-only subclass of the frozen synthetic terrain evaluator."""
from model_terrain import TerrainFarm


class RecordedTerrainFarm(TerrainFarm):
    def __init__(self, *args, **kwargs):
        self.initial_keys = []
        super().__init__(*args, **kwargs)

    def evaluate(self, keys):
        result = super().evaluate(keys)
        if len(self.records) <= 100:
            self.initial_keys.append(result[0].copy())
        return result

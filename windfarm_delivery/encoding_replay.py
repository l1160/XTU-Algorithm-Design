"""Replay completed trajectories against saved metrics; no new physical evaluations."""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "windfarm_research_v4"))
import bootstrap_v4
import numpy as np
import dill
from variants import DiagnosticFarm, Engine, VARIANTS, configure
from campaign_figures import check_complete, table

CATEGORIES = ["identical_keys", "identical_full_order", "feasible_subset_projection", "repair_mapping_collision"]


class ReplayFarm(DiagnosticFarm):
    def set_oracle(self, saved):
        self.saved = saved
        self.cache = {ids.astype("<u2").tobytes(): (metric.copy(), float(cv))
                      for ids, metric, cv in zip(saved["selected"], saved["metrics"], saved["cv"])}
        self.seen_keys, self.seen_orders, self.seen_layouts = set(), set(), set()
        self.key_to_layout, self.order_to_layout = {}, {}
        self.mapping_records = []

    def evaluate(self, keys):
        i = len(self.records)
        canonical = np.clip(np.asarray(keys, dtype=np.float64), 0, 1).copy()
        canonical[canonical == 0] = 0.0  # Normalize signed zero without changing order.
        packed_key = canonical.astype("<f8").tobytes()
        order = np.argsort(canonical, kind="stable").astype("<u2")
        packed_order = order.tobytes()
        raw = np.sort(order[:self.n]).astype("<u2").tobytes()
        # Only known layouts are allowed; ensure_wake raises if a novel layout
        # would otherwise cause a new physical evaluation.
        result = super().evaluate(keys)
        record = self.records[-1]
        final = record[0]
        expected = self.saved["selected"][i].astype("<u2").tobytes()
        assert final == expected, f"Trajectory mismatch at request {i}"
        assert np.array_equal(record[1], self.saved["metrics"][i])
        assert record[2] == self.saved["cv"][i]
        assert record[3] == self.saved["raw_conflict_pairs"][i]
        assert record[4] == self.saved["repair_success"][i]
        assert record[6] == self.saved["repair_replacements"][i]
        repeated = final in self.seen_layouts
        assert repeated == bool(self.saved["cache_hit"][i])
        if packed_key in self.key_to_layout:
            assert self.key_to_layout[packed_key] == final
        if packed_order in self.order_to_layout:
            assert self.order_to_layout[packed_order] == final
        category = "new_layout"
        if repeated:
            if packed_key in self.seen_keys:
                category = CATEGORIES[0]
            elif packed_order in self.seen_orders:
                category = CATEGORIES[1]
            elif record[3] == 0:
                category = CATEGORIES[2]
            else:
                category = CATEGORIES[3]
        self.records[-1] = (*record[:5], repeated, *record[6:])
        self.mapping_records.append(dict(request_index_zero_based=i, category=category,
            raw_feasible=record[3]==0, original_duplicate=repeated,
            keys_sha256=hashlib.sha256(packed_key).hexdigest(),
            full_order_sha256=hashlib.sha256(packed_order).hexdigest(),
            raw_subset_sha256=hashlib.sha256(raw).hexdigest(),
            final_layout_sha256=hashlib.sha256(final).hexdigest()))
        self.seen_keys.add(packed_key); self.seen_orders.add(packed_order); self.seen_layouts.add(final)
        self.key_to_layout[packed_key] = final; self.order_to_layout[packed_order] = final
        return result

    def ensure_wake(self):
        raise RuntimeError("Replay attempted a new physical evaluation; abort")


def run(src, out):
    src, out = Path(src), Path(out)
    cfg, _ = check_complete(src)
    assert cfg["evaluations"] == 2000 and cfg["seeds"] == [201,202,203]
    variants = ["NSGA3_base", "LM_base", "DGEA_base", "DGEA_no_writeback"]
    summaries = []
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for size in (800, 1600):
        for variant in variants:
            for seed in cfg["seeds"]:
                name = f"obs_00_M{size}_N400_{variant}_s{seed}"
                folder = src / name
                runout = out/name
                runout.mkdir(parents=True, exist_ok=True)
                if (runout/"DONE.json").exists():
                    item=json.loads((runout/"DONE.json").read_text(encoding="utf-8"))
                    assert item["replay_source_sha256"]==source_hash
                    summaries.append(item)
                    continue
                with np.load(folder/"evaluations.npz") as a:
                    saved = {k:a[k].copy() for k in a.files}
                options = VARIANTS[variant]
                configure(options)
                checkpoint=runout/"last.pkl"
                if checkpoint.exists():
                    with checkpoint.open("rb") as stream:
                        state=dill.load(stream)
                    assert state["source_sha256"]==source_hash
                    engine=state["engine"];farm=engine.farm
                    print("RESUME REPLAY",name,len(farm.records),flush=True)
                else:
                    farm = ReplayFarm("obs_00", size, 400, options=options)
                    assert np.array_equal(farm.xy, saved["candidate_xy"])
                    farm.set_oracle(saved)
                    engine = Engine(farm, options["method"], seed, cfg["evaluations"], cfg["population"])
                last_checkpoint=len(farm.records)
                while engine.remaining:
                    engine.step()
                    if len(farm.records)-last_checkpoint>=500 or not engine.remaining:
                        temp=checkpoint.with_suffix(".pkl.tmp")
                        with temp.open("wb") as stream:
                            dill.dump(dict(engine=engine,source_sha256=source_hash),stream)
                        temp.replace(checkpoint)
                        last_checkpoint=len(farm.records)
                assert len(farm.records) == 2000
                trace = json.loads((folder/"trace.json").read_text(encoding="utf-8"))
                assert len(trace) == len(engine.trace)
                for a,b in zip(trace, engine.trace):
                    assert a.keys() == b.keys()
                    for key in a:
                        assert np.allclose(a[key], b[key], rtol=1e-12, atol=1e-12), (name,key)
                rr = farm.mapping_records
                counts = {key:sum(r["category"]==key for r in rr) for key in CATEGORIES}
                assert sum(counts.values()) == int(saved["cache_hit"].sum())
                table(runout/"mapping_trace.csv", rr)
                item = dict(run=name,M=size,variant=variant,seed=seed,requests=len(rr),
                    original_duplicates=int(saved["cache_hit"].sum()),**counts,
                    trajectory_and_trace_verified=True,new_physical_evaluations=0,
                    replay_source_sha256=source_hash)
                (runout/"DONE.json").write_text(json.dumps(item,indent=2),encoding="utf-8")
                summaries.append(item)
                print("ENCODING REPLAY",len(summaries),"/24",name,flush=True)
    table(out/"per_run_mapping_counts.csv",summaries)
    aggregate=[]
    for size in (800,1600):
        for variant in variants:
            rr=[r for r in summaries if r["M"]==size and r["variant"]==variant]
            aggregate.append(dict(M=size,variant=variant,runs=len(rr),
                **{k:sum(r[k] for r in rr) for k in ["requests","original_duplicates",*CATEGORIES]}))
    table(out/"aggregate_mapping_counts.csv",aggregate)
    return dict(status="complete",replayed_runs=len(summaries),replayed_requests=48000,
        all_original_layout_sequences_verified=True,all_saved_trace_summaries_verified=True,
        new_optimization_runs=0,new_physical_evaluations=0,
        scope="Deterministic replay with saved objective oracle; immediate collision categories, not total causal attribution.")

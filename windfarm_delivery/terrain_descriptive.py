"""Descriptive interpretation of the complete prespecified terrain screening."""
from pathlib import Path
import hashlib
import json
import statistics
from campaign_figures import check_complete, csvrows, table


def run(src, out):
    src, out = Path(src), Path(out)
    cfg, rows = check_complete(src)
    assert cfg["stage"] == "synthetic_terrain_screening"
    assert len(rows) == 90 and cfg["evaluations"] == 2000
    aggregate = csvrows(src / "analysis/aggregate_metrics.csv")
    assert len(aggregate) == 30
    cases = sorted({r["case"] for r in rows})
    contrasts, rankings = [], []
    for case in cases:
        part = [r for r in rows if r["case"] == case]
        assert len(part) == 15
        indexed = {(r["variant"], int(r["seed"])): r for r in part}
        for treatment, control in [
            ("LM_fill_population", "LM_base"), ("CCwake_base", "NSGA3_base")
        ]:
            differences = []
            for seed in cfg["seeds"]:
                a, b = indexed[treatment, seed], indexed[control, seed]
                assert a["initial_layouts_sha256"] == b["initial_layouts_sha256"]
                differences.append(float(a["HV"]) - float(b["HV"]))
            contrasts.append(dict(
                case=case, treatment=treatment, control=control, n=len(differences),
                mean_HV_difference=statistics.mean(differences),
                min_paired_difference=min(differences), max_paired_difference=max(differences),
                wins=sum(d > 1e-12 for d in differences),
                ties=sum(abs(d) <= 1e-12 for d in differences),
                losses=sum(d < -1e-12 for d in differences),
                seed_differences=json.dumps(dict(zip(cfg["seeds"], differences))),
                interpretation="Descriptive only; three seeds, no significance test"))
        ordered = sorted([r for r in aggregate if r["case"] == case],
                         key=lambda r: -float(r["HV_mean"]))
        for rank, r in enumerate(ordered, 1):
            observed = [float(v["HV"]) for v in part if v["variant"] == r["variant"]]
            assert abs(statistics.mean(observed) - float(r["HV_mean"])) < 1e-12
            assert abs(statistics.stdev(observed) - float(r["HV_sd"])) < 1e-12
            rankings.append(dict(case=case, descriptive_rank=rank, variant=r["variant"],
                                 HV_mean=r["HV_mean"], HV_sd=r["HV_sd"]))
    table(out / "paired_descriptive_comparisons.csv", contrasts)
    table(out / "within_case_descriptive_ranks.csv", rankings)
    lm = [r for r in contrasts if r["treatment"] == "LM_fill_population"]
    cc = [r for r in contrasts if r["treatment"] == "CCwake_base"]
    evidence = dict(
        stage="Complete terrain screening; not long-budget confirmation",
        runs=90, requests=180000, cases=6, seeds=cfg["seeds"],
        post_feasible_rate_min=min(float(r["post_feasible_rate"]) for r in rows),
        lm_fill=dict(cases_with_positive_mean=sum(r["mean_HV_difference"] > 0 for r in lm),
                     paired_wins=sum(r["wins"] for r in lm),
                     paired_losses=sum(r["losses"] for r in lm),
                     comparisons=18,
                     duplicate_rate_mean_range=[
                         min(float(r["duplicate_rate_mean"]) for r in aggregate if r["variant"] == "LM_fill_population"),
                         max(float(r["duplicate_rate_mean"]) for r in aggregate if r["variant"] == "LM_fill_population")]),
        cc_vs_nsga3=dict(
            M800_paired_wins=sum(r["wins"] for r in cc if "_M800_" in r["case"]),
            M1600_paired_wins=sum(r["wins"] for r in cc if "_M1600_" in r["case"]),
            comparisons_per_M=9),
        limitations=[
            "Three seeds per case do not establish confirmatory significance.",
            "Counts across terrains reuse seed labels and must not be treated as 18 independent significance-test replicates.",
            "Candidate count changes candidate resolution; turbine count remains 400.",
            "All HV comparisons are within case, using its common empirical normalization.",
            "Population filling improves this screening mean but does not eliminate mapping duplicates.",
            "Synthetic height and simplified geometric wakes are not calibrated real-terrain predictions."
        ])
    (out / "interpretation.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    source_paths = [src / "frozen_config.json", src / "analysis/per_run_metrics.csv",
                    src / "analysis/aggregate_metrics.csv", Path(__file__)]
    (out / "sources.json").write_text(json.dumps({
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
    }, indent=2), encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False), flush=True)
    return dict(status="complete", runs=90, cases=6, descriptive_contrasts=len(contrasts),
                new_optimization_runs=0, significance_tests=0)

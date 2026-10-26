"""Prespecify a bounded confirmation before observing any new run."""
from pathlib import Path
import argparse
import itertools
from windfarm_research_v6.compat import ROOT
from windfarm_research_v6.run import validate_config
from windfarm_research_v6.lifecycle import sha, now
from .runtime import METHODS, signature, save_json

CASES = [
    dict(id="flat_800", scenario="obs_00", terrain="flat", candidates=800, turbines=400),
    dict(id="flat_1600", scenario="obs_00", terrain="flat", candidates=1600, turbines=400),
    dict(id="ridge_1600", scenario="obs_01", terrain="ridge", candidates=1600, turbines=400),
]
SEEDS = list(range(1101, 1111))


def prepare(output, dependency_path, check_output):
    from .runtime import read
    output = Path(output).resolve()
    if (output / "plan.json").exists():
        raise FileExistsError("A frozen plan exists; use its campaign_job.json to resume")
    if read(Path(check_output) / "DONE.json")["status"] != "passed":
        raise ValueError("Unified harness checks must pass first")
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    jobs = []
    # Pair order and case order rotate independently of optimizer RNG.
    for seed_index, seed in enumerate(SEEDS):
        for case in CASES[seed_index % 3:] + CASES[:seed_index % 3]:
            pairs = [METHODS[i:i+2] for i in (0, 2, 4)]
            pairs = pairs[seed_index % 3:] + pairs[:seed_index % 3]
            if seed_index % 2:
                pairs = [pair[::-1] for pair in pairs]
            for method in itertools.chain.from_iterable(pairs):
                key = f"{case['id']}_s{seed}_{method}"
                config = validate_config(dict(
                    method=method, scenario=case["scenario"], terrain=case["terrain"],
                    candidates=case["candidates"], turbines=case["turbines"], seed=seed,
                    candidate_seed=20261012, population=100, requests=10000, checkpoint_requests=500))
                config_path = output / "configs" / (key + ".json")
                folder = output / "runs" / key
                worker_path = output / "jobs" / (key + ".json")
                save_json(config_path, config)
                worker = dict(mode="Run", output=str(folder), config=str(config_path), dependency_path=dependency_path)
                save_json(worker_path, worker)
                jobs.append(dict(id=key, case=case["id"], method=method, seed=seed,
                                 config=str(config_path), config_sha256=sha(config_path), output=str(folder),
                                 worker_job=str(worker_path), worker_job_sha256=sha(worker_path)))
                sources.update(signature(config)["sources"])
    plan = dict(
        created_at=now(), output=str(output), cases=CASES, seeds=SEEDS, methods=list(METHODS), jobs=jobs,
        source_sha256=sources, workers=1, dependency_path=dependency_path,
        check_output=str(Path(check_output).resolve()), check_sha256=sha(Path(check_output)/"DONE.json"),
        setup="Warm all case geometry and wake caches once before timed cells; setup reported separately. No optimizer requests in warmup.",
        primary_endpoints=["HV at 10000 evaluation requests", "Optimized HV at the paired baseline total worker CPU budget"],
        normalization="One pooled ideal/nadir per case from every feasible evaluated objective vector of all 60 runs; reference [1.1,1.1,1.1]; no cross-case raw-HV comparisons.",
        inference="10 paired seeds per family and case; exact two-sided sign-flip of paired mean difference (1024 assignments), Holm correction across 18 primary comparisons; finite scenario scope, no universal superiority claim.",
        secondary_endpoints=["Full-budget CPU and wall time", "Time to paired baseline final HV, including nonattainment",
                             "Feasibility, repeated layouts, unique feasible layouts, working/archive sizes",
                             "Pareto objectives, gross/effective area, convergence, pairwise dominance coverage"],
        timing="Single worker, BLAS/OpenMP one thread; no additional optimization jobs. CPU excludes child-controller time. Generation CPU traces are measured from farm construction and preserve active-time offsets over full-state resume; interruption/checkpoint overhead is disclosed.",
        fixed_cpu_rule="Last recorded generation whose active CPU <= paired baseline total worker CPU; hold final quality if finished sooner; no interpolation or future information.",
        candidate_scope="Same candidate set within each case and same first 100 layouts within each seed. 800/1600 are separate deterministic candidate discretizations, not claimed nested grids.",
        transfer_scope="ridge_1600 changes wind scenario and synthetic terrain jointly; it tests transfer, not separate causal effects of each factor.",
        acceptance="All 180 cells complete with matching sources, independent saved-result checks, final plots and paired report. A negative comparison is a valid finding and does not trigger automatic tuning or extra runs.",
        limitations=["CCGDE3 is the project's CCGDE3-inspired Python adaptation with wake grouping, not a bitwise reproduction of original MATLAB code.",
                     "Rotor-overlap Jensen, fixed CT, 24 wind sectors, synthetic terrain and approximate economic/area objectives are engineering research proxies.",
                     "Three cases and ten new seeds confirm this bounded benchmark only; they do not establish universal convergence or field-calibrated accuracy."])
    save_json(output / "plan.json", plan)
    job = dict(mode="Compare", output=str(output), plan=str(output / "plan.json"),
               plan_sha256=sha(output / "plan.json"), dependency_path=dependency_path)
    save_json(output / "campaign_job.json", job)
    save_json(output / "progress.json", dict(status="prepared", total_runs=180, completed_runs=0, updated_at=now()))
    return job


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--dependency-path", required=True)
    parser.add_argument("--check-output", required=True)
    args = parser.parse_args()
    print(prepare(args.output, args.dependency_path, args.check_output))

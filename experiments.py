import csv
import sys
from pathlib import Path

from steering_utils import (
    load_gpt2_small, load_sae_from_neuronpedia, Dist,
    ClampOp, CondDistOp, EveryOtherTokenOp, FibonacciTokensOp,
    NthTokenOp, SpecificTokensOp, ThresholdOp, ScaleOp, AddOp, ZeroOp,
)

#import other experiment scripts here as needed
from experiment_scripts.bulk_features import bulk_feature_stats
from experiment_scripts.affectations import affectation_experiment
from experiment_scripts.affectations2 import affectation_experiment_v2
from experiment_scripts.steering_dashboard import SteeringExperiment


def sample_experiment():
    model = load_gpt2_small()
    sae = load_sae_from_neuronpedia("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    model.add_sae(sae)

    sae.clamp(18493, 40.0)
    text = model.generate("The Eiffel Tower is located in", max_tokens=50)
    print(text)

    sae.clear()
    sae.analyze("Hello world", html_output="index.html")


def dashboard_experiment(
    prompt: str = "I like sharks",
    n_tokens: int = 100,
    feature_id: int = 18493,
    output: str = "output/dashboard.html",
):
    """
    Compare multiple steering strategies for a single feature on a single prompt.
    Opens the result as an HTML dashboard at `output`.

    Run:
        python experiments.py dashboard_experiment
        python experiments.py dashboard_experiment prompt "The ocean is" n_tokens 60
    """
    model = load_gpt2_small()
    sae   = load_sae_from_neuronpedia("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    model.add_sae(sae)

    fid = int(feature_id)

    exp = SteeringExperiment(model, feature_ids=[fid])

    # Baseline: no intervention
    exp.add_method("baseline", {})

    # Always clamp to 40
    exp.add_method("clamp_40", {fid: ClampOp(40)})

    # With probability 0.3, sample from N(40, 10)
    exp.add_method("cond_dist_N(40,10)_p0.3", {fid: CondDistOp(0.3, Dist("normal", 40, 10))})

    # Clamp every other token
    exp.add_method("every_other_token", {fid: EveryOtherTokenOp(40)})

    # Clamp on Fibonacci-indexed tokens (0,1,2,3,5,8,13,…)
    exp.add_method("fibonacci_tokens", {fid: FibonacciTokensOp(40)})

    # Clamp every 3rd token
    exp.add_method("every_3rd_token", {fid: NthTokenOp(3, 40)})

    # Clamp only on tokens 1,2,3,5,8,13 (manual list)
    exp.add_method("specific_tokens_1-2-3-5-8-13", {fid: SpecificTokensOp([1,2,3,5,8,13], 40)})

    # Only intervene when activation exceeds 5 naturally (threshold gate)
    exp.add_method("threshold_gate_>5", {fid: ThresholdOp(threshold=5, value=40)})

    # Scale natural activation by 3×
    exp.add_method("scale_3x", {fid: ScaleOp(3.0)})

    # Add 20 to natural activation
    exp.add_method("add_20", {fid: AddOp(20.0)})

    # Zero out the feature entirely
    exp.add_method("zero_out", {fid: ZeroOp()})

    results = exp.run(str(prompt), n_tokens=int(n_tokens))
    results.save_html(str(output))


EXPERIMENTS = {
    "sample": sample_experiment,
    "dashboard_experiment": dashboard_experiment,
    "bulk_feature_stats": bulk_feature_stats,
    "affectation_experiment": affectation_experiment,
    "affectation_experiment_v2": affectation_experiment_v2,
}

def _parse_kwargs(args: list[str]) -> dict:
    """Parse 'key value' pairs from a flat arg list into a dict.

    Tries to coerce each value to int, then float, otherwise keeps it as str.
    Example: ['mode', '1', 'top_n', '5'] -> {'mode': 1, 'top_n': 5}
    """
    kwargs: dict = {}
    it = iter(args)
    for key in it:
        raw = next(it, None)
        if raw is None:
            break
        for cast in (int, float):
            try:
                raw = cast(raw)
                break
            except ValueError:
                pass
        kwargs[key] = raw
    return kwargs


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in EXPERIMENTS:
        print(f"Usage: {sys.argv[0]} <experiment_name> [key value ...]")
        print("Available experiments:")
        for name in EXPERIMENTS:
            print(f"  - {name}")
        sys.exit(1)

    experiment_name = sys.argv[1]
    kwargs = _parse_kwargs(sys.argv[2:])
    EXPERIMENTS[experiment_name](**kwargs)
    print("Experiment complete. Exiting...")
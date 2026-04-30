import csv
import sys
from pathlib import Path

from steering_utils import load_gpt2_small, load_sae_from_neuronpedia

#import other experiment scripts here as needed
from experiment_scripts.bulk_features import bulk_feature_stats
from experiment_scripts.affectations import affectation_experiment
from experiment_scripts.affectations2 import affectation_experiment_v2


def sample_experiment():
    model = load_gpt2_small()
    sae = load_sae_from_neuronpedia("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    model.add_sae(sae)

    sae.clamp(18493, 40.0)
    text = model.generate("The Eiffel Tower is located in", max_tokens=50)
    print(text)

    sae.clear()
    sae.analyze("Hello world", html_output="index.html")


EXPERIMENTS = {
    "sample": sample_experiment,
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
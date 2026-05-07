from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConfigBlock:
    name: str
    values: dict[str, Any]


def diff_blocks(a: ConfigBlock, b: ConfigBlock) -> list[str]:
    keys = sorted(set(a.values) | set(b.values))
    diffs: list[str] = []
    for key in keys:
        a_val = a.values.get(key, "<missing>")
        b_val = b.values.get(key, "<missing>")
        if a_val != b_val:
            diffs.append(f"- {key}: notebook={a_val!r} | website={b_val!r}")
    return diffs


NOTEBOOK = ConfigBlock(
    name="affectations2_hot_cold.ipynb",
    values={
        "model_name": "google/gemma-3-270m-it",
        "sae_path": "saved_saes/shrink_mid_1/shrink/best_step_update",
        "device": "cuda:2 (if available, otherwise cpu)",
        "dtype": "bfloat16 (if cuda else float32)",
        "prompt_format": "gemma-it chat template",
        "prompt": "is the temperature outside hot or cold?",
        "steering_mode": "notebook_hot_cold",
        "vector_source": "saved hot_cold_delta_avg.npy",
        "token_index": "all",
        "projector_location": "post_activation",
        "do_sample": False,
        "use_past_kv_cache": False,
        "top_p": 0.95,
        "temperature": 0.8,
        "max_new_tokens": 100,
        "seed": 42,
        "top_k": 1000,
        "method": "add hot_vector / subtract hot_vector",
        "factor": 10,
        "baseline_eos_override": False,
        "strip_end_of_turn": False,
    },
)

WEBSITE = ConfigBlock(
    name="web_sae_playground.py",
    values={
        "model_name": "google/gemma-3-270m-it",
        "sae_path": "saved_saes/shrink_mid_1/shrink/best_step_update",
        "device": "cuda:2 (if launched with --device cuda:2, otherwise server arg)",
        "dtype": "bfloat16 (if cuda else float32)",
        "prompt_format": "gemma-it chat template",
        "prompt": "is the temperature outside hot or cold?",
        "steering_mode": "notebook_hot_cold",
        "vector_source": "saved hot_cold_delta_avg.npy",
        "token_index": "all",
        "projector_location": "post_activation",
        "do_sample": False,
        "use_past_kv_cache": False,
        "top_p": 0.95,
        "temperature": 0.8,
        "max_new_tokens": 100,
        "seed": 42,
        "top_k": 1000,
        "method": "add hot_vector / subtract hot_vector",
        "factor": 10,
        "baseline_eos_override": False,
        "strip_end_of_turn": False,
    },
)


def main() -> None:
    print(f"Comparing {NOTEBOOK.name} vs {WEBSITE.name}\n")
    diffs = diff_blocks(NOTEBOOK, WEBSITE)
    if not diffs:
        print("No differences detected in the recorded defaults.")
        return
    print("Differences:")
    for line in diffs:
        print(line)


if __name__ == "__main__":
    main()

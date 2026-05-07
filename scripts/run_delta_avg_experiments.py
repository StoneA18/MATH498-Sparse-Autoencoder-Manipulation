from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ensure relative paths and imports resolve from the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if Path.cwd() != PROJECT_ROOT:
    try:
        import os

        os.chdir(PROJECT_ROOT)
    except OSError:
        pass

from trainable_sae import HookPointSpec, SAEConnector, TrainableSAE, load_hooked_transformer, resolve_device
SAVED_SAES_ROOT = PROJECT_ROOT / "saved_saes"
NOTEBOOK_MODEL_NAME = "google/gemma-3-270m-it"
NOTEBOOK_SAE_PATH = PROJECT_ROOT / "saved_saes/shrink_mid_1/shrink/best_step_update"
NOTEBOOK_HOT_WORDS = ["hot"]
NOTEBOOK_COLD_WORDS = ["cold"]

AFFECTATIONS_DATA = PROJECT_ROOT / "experiment_scripts/experiment_data/affectations"
AFFECTATIONS2_DATA = PROJECT_ROOT / "experiment_scripts/experiment_data/affectations2"

HAPPY_TEXT_PATH = AFFECTATIONS_DATA / "very_happy.txt"
SAD_TEXT_PATH = AFFECTATIONS_DATA / "very_sad.txt"
FIB_SENTENCES_PATH = AFFECTATIONS2_DATA / "fib_sentences.txt"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    positive_label: str
    negative_label: str


EXPERIMENTS = (
    ExperimentSpec(name="hot_cold", positive_label="hot", negative_label="cold"),
    ExperimentSpec(name="happy_sad", positive_label="happy", negative_label="sad"),
)


def _first_scalar(value: Any, default: Any) -> Any:
    if value is None:
        return default
    while isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return value.detach().reshape(-1)[0].cpu().item()
    return value


def _read_word_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing word list: {path}")
    words = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [word for word in words if word]


def _extract_assistant_blocks(text: str) -> list[str]:
    responses: list[str] = []
    current: list[str] = []
    in_assistant = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("**Assistant:**") or stripped.startswith("Assistant:"):
            if in_assistant and current:
                responses.append(" ".join(current).strip())
                current = []
            in_assistant = True
            content = stripped.replace("**Assistant:**", "").replace("Assistant:", "").strip()
            if content:
                current.append(content)
            continue
        if (
            stripped.startswith("**User:**")
            or stripped.startswith("User:")
            or stripped.startswith("Section")
            or stripped.startswith("SECTION")
        ):
            if in_assistant and current:
                responses.append(" ".join(current).strip())
                current = []
            in_assistant = False
            continue
        if in_assistant and stripped:
            current.append(stripped)

    if in_assistant and current:
        responses.append(" ".join(current).strip())
    return responses


def _build_hot_cold_sentences() -> tuple[list[str], list[str]]:
    lines = [
        line.strip()
        for line in FIB_SENTENCES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    hot_sentences: list[str] = []
    cold_sentences: list[str] = []

    for line in lines:
        body = " ".join(line.split()[1:])
        blank_idx = body.index("<blank>")
        prefix, suffix = body[:blank_idx], body[blank_idx + len("<blank>"):]
        for hot_word in NOTEBOOK_HOT_WORDS:
            hot_sentences.append(prefix + hot_word + suffix)
        for cold_word in NOTEBOOK_COLD_WORDS:
            cold_sentences.append(prefix + cold_word + suffix)

    return hot_sentences, cold_sentences


def _build_happy_sad_sentences() -> tuple[list[str], list[str]]:
    happy_text = HAPPY_TEXT_PATH.read_text(encoding="utf-8")
    sad_text = SAD_TEXT_PATH.read_text(encoding="utf-8")

    happy_responses = _extract_assistant_blocks(happy_text)
    sad_responses = _extract_assistant_blocks(sad_text)

    return happy_responses, sad_responses


def _encode_sentence(
    sentence: str,
    model: Any,
    connector: SAEConnector,
    sae: TrainableSAE,
    device: str,
) -> torch.Tensor:
    tokens = model.to_tokens(sentence).to(device)
    with torch.no_grad():
        acts = connector.collect_activations(tokens)
        feats = sae.encode(acts)
    return feats.reshape(-1, sae.d_sae).to(dtype=torch.float32)


def _average_activations(
    sentences: list[str],
    model: Any,
    connector: SAEConnector,
    sae: TrainableSAE,
    device: str,
) -> torch.Tensor:
    if not sentences:
        raise ValueError("No sentences provided.")
    sums = torch.zeros(sae.d_sae, device=device, dtype=torch.float32)
    count = 0
    for sentence in sentences:
        feats = _encode_sentence(sentence, model, connector, sae, device)
        sums += feats.mean(dim=0)
        count += 1
    return (sums / max(count, 1)).detach().cpu()


def _resolve_hook_point(model: Any, sae: TrainableSAE) -> str:
    return str(
        _first_scalar(
            sae.cfg.metadata.get(
                "hook_point",
                HookPointSpec(layer=model.cfg.n_layers // 2, site="resid_post").name,
            ),
            HookPointSpec(layer=model.cfg.n_layers // 2, site="resid_post").name,
        )
    )


def _resolve_model_name(sae: TrainableSAE) -> str:
    return str(_first_scalar(sae.cfg.metadata.get("model_name", "google/gemma-3-270m-it"), ""))


def _discover_sae_dirs(root: Path) -> list[Path]:
    return sorted({checkpoint.parent for checkpoint in root.glob("**/trainable_sae.pt")})


def _save_outputs(
    output_dir: Path,
    name: str,
    delta_avg: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{name}_delta_avg.npy", delta_avg)
    (output_dir / f"{name}_delta_avg.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _notebook_hot_cold_delta_avg(
    hot_sentences: list[str],
    cold_sentences: list[str],
    model: Any,
    connector: SAEConnector,
    sae: TrainableSAE,
    device: str,
) -> torch.Tensor:
    hot_sums = torch.zeros(sae.d_sae, device=device, dtype=torch.float32)
    cold_sums = torch.zeros(sae.d_sae, device=device, dtype=torch.float32)
    hot_counts = torch.zeros(sae.d_sae, device=device, dtype=torch.float32)
    cold_counts = torch.zeros(sae.d_sae, device=device, dtype=torch.float32)

    for hot_sentence in hot_sentences:
        hot_feats = _encode_sentence(hot_sentence, model, connector, sae, device)
        hot_sums += hot_feats.mean(dim=0)
        hot_counts += 1

    for cold_sentence in cold_sentences:
        cold_feats = _encode_sentence(cold_sentence, model, connector, sae, device)
        cold_sums += cold_feats.mean(dim=0)
        cold_counts += 1

    avg_hot = (hot_sums / hot_counts).detach().cpu()
    avg_cold = (cold_sums / cold_counts).detach().cpu()
    delta_avg = avg_hot - avg_cold

    top_k_values, _ = torch.topk(torch.abs(delta_avg), k=1000)
    min_top_k = top_k_values[-1]
    delta_avg[torch.abs(delta_avg) < min_top_k] = 0
    return delta_avg


def run_experiments(
    device: str,
    local_files_only: bool,
    include_happy_sad: bool,
    notebook_sae_only: bool,
) -> None:
    if notebook_sae_only:
        sae_dirs = [NOTEBOOK_SAE_PATH]
    else:
        sae_dirs = _discover_sae_dirs(SAVED_SAES_ROOT)
        if not sae_dirs:
            raise FileNotFoundError(f"No trainable_sae.pt found under {SAVED_SAES_ROOT}")

    resolved_device = resolve_device(device)

    model_cache: dict[str, Any] = {}

    for sae_dir in sae_dirs:
        sae = TrainableSAE.load(sae_dir, device=resolved_device)
        sae.eval()

        model_name = NOTEBOOK_MODEL_NAME if notebook_sae_only else _resolve_model_name(sae)
        if model_name not in model_cache:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model_cache[model_name] = load_hooked_transformer(
                model_name,
                device=resolved_device,
                dtype=dtype,
                local_files_only=local_files_only,
            )
            model_cache[model_name].to(dtype)
        model = model_cache[model_name]

        hook_point = _resolve_hook_point(model, sae)
        connector = SAEConnector(
            model=model,
            sae=sae,
            hook_point=hook_point,
            device=resolved_device,
            preserve_error=True,
        )

        print(f"\n=== {sae_dir} ===")
        print(f"model:      {model_name}")
        print(f"hook point: {hook_point}")
        print(f"d_sae:      {sae.d_sae}")

        output_dir = sae_dir / "experiments"
        specs = EXPERIMENTS if include_happy_sad else (EXPERIMENTS[0],)

        for spec in specs:
            if spec.name == "hot_cold":
                positives, negatives = _build_hot_cold_sentences()
                source_detail = {
                    "fib_sentences": str(FIB_SENTENCES_PATH.relative_to(PROJECT_ROOT)),
                    "hot_words": NOTEBOOK_HOT_WORDS,
                    "cold_words": NOTEBOOK_COLD_WORDS,
                }
            elif spec.name == "happy_sad":
                positives, negatives = _build_happy_sad_sentences()
                source_detail = {
                    "happy_text": str(HAPPY_TEXT_PATH.relative_to(PROJECT_ROOT)),
                    "sad_text": str(SAD_TEXT_PATH.relative_to(PROJECT_ROOT)),
                }
            else:
                raise ValueError(f"Unknown experiment: {spec.name}")

            if spec.name == "hot_cold":
                delta_avg = _notebook_hot_cold_delta_avg(
                    positives,
                    negatives,
                    model,
                    connector,
                    sae,
                    resolved_device,
                ).numpy()
            else:
                avg_pos = _average_activations(positives, model, connector, sae, resolved_device)
                avg_neg = _average_activations(negatives, model, connector, sae, resolved_device)
                delta_avg = (avg_pos - avg_neg).numpy()

            metadata = {
                "experiment": spec.name,
                "positive_label": spec.positive_label,
                "negative_label": spec.negative_label,
                "model_name": model_name,
                "hook_point": hook_point,
                "sae_path": str(sae_dir.relative_to(PROJECT_ROOT)),
                "d_sae": int(sae.d_sae),
                "positive_count": len(positives),
                "negative_count": len(negatives),
                "sources": source_detail,
            }

            _save_outputs(output_dir, spec.name, delta_avg, metadata)
            print(
                f"Saved {spec.name} delta_avg -> {output_dir / (spec.name + '_delta_avg.npy')}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute notebook-parity delta_avg vectors for every SAE in saved_saes."
        )
    )
    parser.add_argument(
        "--device",
        default="cuda:2" if torch.cuda.is_available() else "cpu",
        help="Torch device. Default matches the notebook: cuda:2 if CUDA is available, else cpu.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use local model files (do not download).",
    )
    parser.add_argument(
        "--include-happy-sad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compute happy/sad. Enabled by default; pass --no-include-happy-sad to skip it.",
    )
    parser.add_argument(
        "--notebook-sae-only",
        action="store_true",
        help="Only run the single SAE used in affectations2_hot_cold.ipynb.",
    )
    parser.add_argument(
        "--all-saes",
        action="store_true",
        help="Deprecated no-op: all saved_saes checkpoints are processed by default.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(
        device=args.device,
        local_files_only=args.local_files_only,
        include_happy_sad=args.include_happy_sad,
        notebook_sae_only=args.notebook_sae_only,
    )

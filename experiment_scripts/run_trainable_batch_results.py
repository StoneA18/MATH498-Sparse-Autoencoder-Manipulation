from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sae_generation import (  # noqa: E402
    DEFAULT_SAE_ROOTS,
    GemmaSaeGenerator,
    PresetGenerationConfig,
    best_step_update_options,
    notebook_vector_defaults,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiment_scripts" / "trainable_results"
EXPERIMENTS = ("hot_cold", "happy_sad")
METHOD_FACTORS = {
    "add": 15.0,
    "project": 1.0,
}
CSV_FIELDS = (
    "sae_path",
    "prompt_index",
    "prompt",
    "experiment",
    "variable_1",
    "variable_2",
    "steering_method",
    "factor",
    "baseline_response",
    "response_1",
    "response_2",
    "error",
)


def project_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_prompts(path: Path) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"No non-empty prompts found in {path}")
    return prompts


def default_output_path(output_dir: Path, prompt_file: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = prompt_file.stem or "prompts"
    return output_dir / f"trainable_results_{stem}_{timestamp}.csv"


def success_rows(
    *,
    result: dict[str, Any],
    prompt_index: int,
    prompt: str,
    option: Any,
    experiment: str,
    method: str,
    factor: float,
) -> list[dict[str, Any]]:
    return [
        {
            "sae_path": project_path(option.path),
            "prompt_index": prompt_index,
            "prompt": prompt,
            "experiment": experiment,
            "variable_1": result.get("positiveLabel", ""),
            "variable_2": result.get("negativeLabel", ""),
            "steering_method": method,
            "factor": factor,
            "baseline_response": result.get("baseline", ""),
            "response_1": result.get("positiveSteered", ""),
            "response_2": result.get("negativeSteered", ""),
            "error": "",
        }
    ]


def error_rows(
    *,
    error: Exception,
    prompt_index: int,
    prompt: str,
    option: Any,
    experiment: str,
    method: str,
    factor: float,
    baseline: str = "",
) -> list[dict[str, Any]]:
    _, positive_label, negative_label, _, _ = notebook_vector_defaults(experiment)
    return [
        {
            "sae_path": project_path(option.path),
            "prompt_index": prompt_index,
            "prompt": prompt,
            "experiment": experiment,
            "variable_1": positive_label,
            "variable_2": negative_label,
            "steering_method": method,
            "factor": factor,
            "baseline_response": baseline,
            "response_1": "",
            "response_2": "",
            "error": str(error),
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CSV results for every saved TrainableSAE using preset steering vectors."
    )
    parser.add_argument(
        "prompt_file",
        type=Path,
        help="Text file with one prompt per non-empty line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to a timestamped file inside --output-dir.",
    )
    parser.add_argument(
        "--sae-root",
        action="append",
        type=Path,
        help="Directory to scan for trainable_sae.pt checkpoints. Can be passed more than once.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--use-past-kv-cache", action="store_true")
    parser.add_argument("--top-k", type=int, default=None, help="Override experiment default top-k.")
    parser.add_argument(
        "--sae-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run with SAE reconstruction and steering enabled.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed generation instead of recording error rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_file = args.prompt_file.expanduser().resolve()
    prompts = read_prompts(prompt_file)
    output_dir = args.output_dir.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(output_dir, prompt_file)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sae_roots = tuple(args.sae_root) if args.sae_root else DEFAULT_SAE_ROOTS
    generator = GemmaSaeGenerator(
        sae_roots=sae_roots,
        device=args.device,
        model_dtype=args.model_dtype,
        local_files_only=args.local_files_only,
    )
    all_options = generator.runtime.options
    options = best_step_update_options(all_options)
    if not options:
        roots = ", ".join(str(root) for root in sae_roots)
        raise FileNotFoundError(
            "No trainable_sae.pt checkpoints in best_step_update folders found "
            f"under: {roots}"
        )

    print(f"Loaded {len(prompts)} prompt(s) from {prompt_file}")
    print(f"Using {len(options)} best_step_update SAE checkpoint(s) from {len(all_options)} total")
    print(f"Writing {output_path}")

    baseline_cache: dict[tuple[Any, ...], str] = {}
    baseline_errors: dict[tuple[Any, ...], Exception] = {}

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for option in options:
            print(f"\n=== {project_path(option.path)} ===")
            for prompt_index, prompt in enumerate(prompts):
                baseline_config = PresetGenerationConfig(
                    sae_path=option.path,
                    sae_enabled=args.sae_enabled,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                    do_sample=args.do_sample,
                    use_past_kv_cache=args.use_past_kv_cache,
                )
                baseline_key = (
                    option.model_name,
                    prompt,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_p,
                    args.seed,
                    args.do_sample,
                    args.use_past_kv_cache,
                )
                for experiment in EXPERIMENTS:
                    for method, factor in METHOD_FACTORS.items():
                        print(
                            f"prompt {prompt_index + 1}/{len(prompts)} | "
                            f"{experiment} | {method} factor={factor}"
                        )
                        config = PresetGenerationConfig(
                            sae_path=option.path,
                            experiment=experiment,
                            method=method,
                            factor=factor,
                            top_k=args.top_k,
                            direction="both",
                            sae_enabled=args.sae_enabled,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            seed=args.seed,
                            do_sample=args.do_sample,
                            use_past_kv_cache=args.use_past_kv_cache,
                        )
                        try:
                            if baseline_key in baseline_errors:
                                raise baseline_errors[baseline_key]
                            if baseline_key not in baseline_cache:
                                print(f"prompt {prompt_index + 1}/{len(prompts)} | baseline")
                                try:
                                    baseline_cache[baseline_key] = generator.generate_baseline(
                                        prompt,
                                        baseline_config,
                                    )
                                except Exception as baseline_exc:
                                    baseline_errors[baseline_key] = baseline_exc
                                    raise
                            result = generator.generate(
                                prompt,
                                config,
                                baseline=baseline_cache[baseline_key],
                            )
                            rows = success_rows(
                                result=result,
                                prompt_index=prompt_index,
                                prompt=prompt,
                                option=option,
                                experiment=experiment,
                                method=method,
                                factor=factor,
                            )
                        except Exception as exc:
                            if args.fail_fast:
                                raise
                            rows = error_rows(
                                error=exc,
                                prompt_index=prompt_index,
                                prompt=prompt,
                                option=option,
                                experiment=experiment,
                                method=method,
                                factor=factor,
                                baseline=baseline_cache.get(baseline_key, ""),
                            )

                        writer.writerows(rows)
                        handle.flush()

    print(f"\nSaved CSV results to {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_metadata_version = importlib.metadata.version


def _version_with_torch_fallback(package_name: str) -> str:
    version = _metadata_version(package_name)
    if package_name == "torch" and version is None:
        return torch.__version__.split("+", maxsplit=1)[0]
    return version


importlib.metadata.version = _version_with_torch_fallback

from trainable_sae import HookPointSpec, SAEConfig, TrainableSAE, load_hooked_transformer, resolve_device


SAE_VARIANTS = (
    # {"name": "relu", "activation": "relu", "l1_coefficient": 250.},
    {"name": "shrink", "activation": "shrink", "l1_coefficient": 400},
    {"name": "topk", "activation": "topk", "l1_coefficient": 0.0},
    {"name": "topbottomk", "activation": "tbk", "l1_coefficient": 0.0},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ReLU, TopK, TopBottomK, and shrink SAEs.")
    parser.add_argument("--model", default="google/gemma-3-270m-it")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--questions-path", type=Path, default=PROJECT_ROOT / "samples/questions_train.txt")
    parser.add_argument("--question-repeats", type=int, default=3)
    parser.add_argument("--token-budget", type=int, default=100_000_000)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--model-forward-texts", type=int, default=128)
    parser.add_argument("--batch-tokens", type=int, default=8192)
    parser.add_argument("--expansion-factor", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--shrink-threshold", type=float, default=.5)
    parser.add_argument(
        "--pre-layer-norm",
        action="store_true",
        default=False,
        help="Apply non-affine LayerNorm to activations before the SAE encoder.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--sae-dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-root", type=Path, default=PROJECT_ROOT / "custom_saes")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--log-nonzero-features",
        action="store_true",
        default=True,
        help="Record and print the average number of non-zero SAE features per activation token.",
    )
    parser.add_argument(
        "--target-nonzero-features",
        type=float,
        default=None,
        help="Target average number of non-zero features per activation token for ReLU/shrink SAEs.",
    )
    parser.add_argument(
        "--sparsity-control-rate",
        type=float,
        default=0.05,
        help="Feedback update rate for target sparsity control.",
    )
    parser.add_argument(
        "--sparsity-ema-beta",
        type=float,
        default=0.95,
        help="EMA smoothing factor for measured average non-zero features.",
    )
    return parser.parse_args()


def read_questions(path: Path, repeats: int) -> list[str]:
    if repeats <= 0 or not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines * repeats


def iter_dataset_texts(dataset_name: str, dataset_config: Optional[str], split: str) -> Iterator[str]:
    from datasets import load_dataset

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True)

    for row in dataset:
        text = row.get("text", "")
        if text and text.strip():
            yield text


def batched(iterable: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def chain_questions_then_dataset(
    questions: list[str],
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
) -> Iterator[str]:
    shuffled_questions = list(questions)
    random.shuffle(shuffled_questions)
    yield from shuffled_questions
    yield from iter_dataset_texts(dataset_name, dataset_config, split)


def collect_activations_for_text_batch(
    model,
    text_batch: list[str],
    hook_point: str,
    d_in: int,
    device: str,
    context_size: int,
) -> torch.Tensor:
    encoded = model.tokenizer(
        text_batch,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=context_size,
        return_tensors="pt",
    )
    tokens = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=hook_point,
            attention_mask=attention_mask,
        )

    acts = cache[hook_point].detach().to(dtype=torch.float32)
    return acts[attention_mask.bool()].reshape(-1, d_in)


def activation_batches(
    model,
    texts: Iterable[str],
    hook_point: str,
    d_in: int,
    device: str,
    batch_tokens: int,
    model_forward_texts: int,
    context_size: int,
) -> Iterator[torch.Tensor]:
    buffer: list[torch.Tensor] = []
    buffered = 0

    for text_batch in batched(texts, model_forward_texts):
        acts = collect_activations_for_text_batch(
            model=model,
            text_batch=text_batch,
            hook_point=hook_point,
            d_in=d_in,
            device=device,
            context_size=context_size,
        )
        buffer.append(acts)
        buffered += acts.shape[0]

        while buffered >= batch_tokens:
            joined = torch.cat(buffer, dim=0)
            yield joined[:batch_tokens]
            remainder = joined[batch_tokens:]
            buffer = [remainder] if remainder.numel() else []
            buffered = remainder.shape[0] if remainder.numel() else 0

    if buffer:
        yield torch.cat(buffer, dim=0)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr: float,
):
    warmup_steps = min(warmup_steps, max(0, total_steps - 1))
    cosine_steps = max(1, total_steps - warmup_steps)
    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=min_lr,
        )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=min_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )


def build_sae(
    args: argparse.Namespace,
    variant: dict[str, object],
    d_in: int,
    hook_point: str,
    device: str,
    run_dir: Path,
):
    total_steps = args.max_steps or max(1, args.token_budget // args.batch_tokens)

    cfg = SAEConfig(
        d_in=d_in,
        d_sae=d_in * args.expansion_factor,
        activation=str(variant["activation"]),
        k=args.top_k,
        shrink_threshold=args.shrink_threshold,
        pre_layer_norm=args.pre_layer_norm,
        lr=args.lr,
        l1_coefficient=float(variant["l1_coefficient"]),
        dtype=args.sae_dtype,
        device=device,
        metadata={
            "variant": variant["name"],
            "model_name": args.model,
            "hook_point": hook_point,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "token_budget": args.token_budget,
            "context_size": args.context_size,
            "model_forward_texts": args.model_forward_texts,
            "batch_tokens": args.batch_tokens,
        },
    )
    sae = TrainableSAE(cfg, device=device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=cfg.lr)
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_steps, args.min_lr)

    variant_dir = run_dir / str(variant["name"])
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return sae, optimizer, scheduler, variant_dir


def save_one(variant_dir: Path, sae: TrainableSAE, metrics: list[dict[str, float]]) -> None:
    sae.save(variant_dir)
    (variant_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def free_sae_memory(*objects) -> None:
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_target_sparsity_udf(
    target_nonzero_features: Optional[float],
    control_rate: float,
    ema_beta: float,
) -> Optional[Callable[[TrainableSAE, dict[str, float]], None]]:
    if target_nonzero_features is None:
        return None
    if target_nonzero_features <= 0:
        raise ValueError("--target-nonzero-features must be positive.")
    if not 0 <= ema_beta < 1:
        raise ValueError("--sparsity-ema-beta must be in [0, 1).")

    ema_nonzero: Optional[float] = None

    def target_sparsity_udf(sae: TrainableSAE, metrics: dict[str, float]) -> None:
        nonlocal ema_nonzero
        avg_nonzero = metrics["avg_nonzero_features"]
        ema_nonzero = avg_nonzero if ema_nonzero is None else ema_beta * ema_nonzero + (1 - ema_beta) * avg_nonzero
        error = (ema_nonzero - target_nonzero_features) / target_nonzero_features
        multiplier = float(torch.exp(torch.tensor(control_rate * error)).item())

        activation = sae.cfg.activation.lower()
        if activation == "relu":
            sae.cfg.l1_coefficient = max(0.0, sae.cfg.l1_coefficient * multiplier)
            metrics["l1_coefficient"] = float(sae.cfg.l1_coefficient)
        elif activation in ("shrink", "softshrink", "soft_shrink", "soft_threshold"):
            new_threshold = max(0.0, sae.cfg.shrink_threshold * multiplier)
            sae.cfg.shrink_threshold = new_threshold
            if isinstance(sae.activation_fn, torch.nn.Module) and hasattr(sae.activation_fn, "threshold"):
                sae.activation_fn.threshold = float(new_threshold)
            metrics["shrink_threshold"] = float(new_threshold)

        metrics["target_nonzero_features"] = float(target_nonzero_features)
        metrics["ema_nonzero_features"] = float(ema_nonzero)
        metrics["sparsity_error"] = float(error)

    return target_sparsity_udf




def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    model_dtype = getattr(torch, args.model_dtype)
    run_name = args.run_name or f"gemma3_270m_four_saes_{int(time.time())}"
    run_dir = args.save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print(f"Loading model {args.model} on {device} ({args.model_dtype})")
    model = load_hooked_transformer(args.model, device=device, dtype=model_dtype)
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    mid_layer = model.cfg.n_layers // 2
    hook_point = HookPointSpec(layer=mid_layer, site="resid_post").name
    d_in = model.cfg.d_model
    print(f"hook_point={hook_point}, d_in={d_in}")

    questions = read_questions(args.questions_path, args.question_repeats)

    print(f"Training variants sequentially: {', '.join(str(v['name']) for v in SAE_VARIANTS)}")
    print(f"Saving to {run_dir}")

    for variant in SAE_VARIANTS:
        name = str(variant["name"])
        print(f"\n=== Training {name} SAE ===")
        sae, optimizer, scheduler, variant_dir = build_sae(
            args=args,
            variant=variant,
            d_in=d_in,
            hook_point=hook_point,
            device=device,
            run_dir=run_dir,
        )
        metrics: list[dict[str, float]] = []
        texts = chain_questions_then_dataset(questions, args.dataset, args.dataset_config, args.split)
        total_seen_tokens = 0
        step = 0
        sparsity_udf = make_target_sparsity_udf(
            args.target_nonzero_features,
            args.sparsity_control_rate,
            args.sparsity_ema_beta,
        )

        for step, batch in enumerate(
            activation_batches(
                model=model,
                texts=texts,
                hook_point=hook_point,
                d_in=d_in,
                device=device,
                batch_tokens=args.batch_tokens,
                model_forward_texts=args.model_forward_texts,
                context_size=args.context_size,
            ),
            start=1,
        ):
            batch = batch.to(device=device, dtype=torch.float32)
            total_seen_tokens += batch.shape[0]

            step_metrics = sae.training_step(
                batch,
                optimizer,
                record_nonzero_features=args.log_nonzero_features,
                sparsity_udf=sparsity_udf,
            )
            scheduler.step()
            step_metrics["lr"] = scheduler.get_last_lr()[0]
            metrics.append(step_metrics)

            if step == 1 or step % args.log_every == 0:
                nonzero_text = ""
                if args.log_nonzero_features:
                    nonzero_text = f" | avg_nonzero={step_metrics['avg_nonzero_features']:.2f}"
                if sparsity_udf is not None:
                    nonzero_text += f" | ema_nonzero={step_metrics['ema_nonzero_features']:.2f}"
                    if "l1_coefficient" in step_metrics:
                        nonzero_text += f" | l1_coeff={step_metrics['l1_coefficient']:.2e}"
                    if "shrink_threshold" in step_metrics:
                        nonzero_text += f" | shrink_threshold={step_metrics['shrink_threshold']:.4f}"
                print(
                    f"{name} | step {step:05d} | tokens={total_seen_tokens:,} | "
                    f"loss={step_metrics['loss']:.4f} | mse={step_metrics['mse']:.4f} | "
                    f"l1={step_metrics['l1']:.4f}{nonzero_text} | lr={step_metrics['lr']:.2e}"
                )

            if args.max_steps is not None and step >= args.max_steps:
                break
            if total_seen_tokens >= args.token_budget:
                break

        save_one(variant_dir, sae, metrics)
        print(f"Saved {name} SAE to {variant_dir}")
        print(f"Finished {step:,} steps over {total_seen_tokens:,} activation tokens for {name}.")
        free_sae_memory(sae, optimizer, scheduler, metrics)

    print(f"\nSaved all four SAEs under {run_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import torch
import torch.nn.functional as F

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

from trainable_sae import (
    HookPointSpec,
    SAEConfig,
    TrainableSAE,
    load_hooked_transformer,
    resolve_device,
)


@dataclass
class ActivationBatch:
    activations: Optional[torch.Tensor] = None
    tokens: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None

    @property
    def token_count(self) -> int:
        if self.attention_mask is None:
            if self.activations is None:
                if self.tokens is None:
                    return 0
                return int(self.tokens.numel())
            return int(self.activations.shape[0])
        return int(self.attention_mask.sum().item())


SAE_VARIANTS = (
    # {"name": "relu", "activation": "relu", "l1_coef": 250.},
    # {"name": "shrink", "activation": "shrink", "l1_coef": 400},
    # {"name": "topk", "activation": "topk", "l1_coef": 0.0},
    # {"name": "topbottomk", "activation": "tbk", "l1_coef": 0.0},
)

DEFAULT_L1_BY_ACTIVATION = {
    "relu": 250.0,
    "gelu": 1e-4,
    "sigmoid": 1e-4,
    "tanh": 1e-4,
    "identity": 1e-4,
    "shrink": 400.0,
    "softshrink": 400.0,
    "soft_shrink": 400.0,
    "soft_threshold": 400.0,
    "topk": 0.0,
    "tbk": 0.0,
    "topbottomk": 0.0,
    "top_bottom_k": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train configurable SAEs.")
    parser.add_argument("--model", default="google/gemma-3-270m-it")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--questions-path", type=Path, default=PROJECT_ROOT / "samples/questions_train.txt")
    parser.add_argument("--question-repeats", type=int, default=3)
    parser.add_argument(
        "--activations",
        nargs="+",
        default=None,
        choices=tuple(DEFAULT_L1_BY_ACTIVATION),
        help=(
            "Built-in activation names to train, e.g. --activations relu shrink tbk. "
            "When omitted, uses the script's SAE_VARIANTS defaults."
        ),
    )
    parser.add_argument("--token-budget", type=int, default=100_000_000)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--model-forward-texts", type=int, default=128)
    parser.add_argument("--batch-tokens", type=int, default=1024)
    parser.add_argument(
        "--hook-layer",
        default="middle",
        help=(
            "Transformer block layer for SAE activations: 'middle', 'last'/'end', "
            "or an explicit zero-indexed layer number. Ignored when --hook-index "
            "or --hook-point is set."
        ),
    )
    parser.add_argument(
        "--hook-index",
        type=int,
        default=None,
        help=(
            "Explicit zero-indexed Transformer block layer for SAE activations. "
            "Equivalent to --hook-layer <index>, but clearer for scripts."
        ),
    )
    parser.add_argument(
        "--hook-site",
        default="resid_post",
        choices=("resid_pre", "resid_post", "mlp_out", "attn_out"),
        help="TransformerLens hook site within the selected block.",
    )
    parser.add_argument(
        "--hook-point",
        default=None,
        help=(
            "Full TransformerLens hook name to train on, e.g. "
            "blocks.0.hook_resid_pre. Overrides --hook-layer/--hook-index "
            "and --hook-site."
        ),
    )
    parser.add_argument("--expansion-factor", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--k-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Linearly warm TopK/TBK k from a larger initial value down to --top-k.",
    )
    parser.add_argument(
        "--k-warmup-start-frac",
        type=float,
        default=0.5,
        help="Initial k as a fraction of d_sae when --k-warmup is enabled.",
    )
    parser.add_argument(
        "--k-warmup-training-frac",
        type=float,
        default=0.20,
        help="Fraction of total training steps over which k reaches --top-k.",
    )
    parser.add_argument(
        "--shrink-threshold",
        type=float,
        default=.5,
        help="Target soft-threshold for shrink SAEs, warmed from 0 over the first 20%% of training.",
    )
    parser.add_argument(
        "--loss-objective",
        default="reconstruction",
        choices=("reconstruction", "cross_entropy"),
        help=(
            "Train on SAE reconstruction MSE or on final model next-token cross entropy "
            "with the SAE inserted at the hook point."
        ),
    )
    parser.add_argument(
        "--l1-context-coef",
        type=float,
        default=0.0,
        help=(
            "Context feature reuse penalty coefficient. When >0, batches preserve "
            "the text/token axis and penalize repeated use of the same feature "
            "across words in a context."
        ),
    )
    parser.add_argument(
        "--l1-coef",
        type=float,
        default=None,
        help=(
            "Override the per-token feature L1 coefficient for every trained SAE. "
            "When omitted, each variant uses its built-in default."
        ),
    )
    parser.add_argument(
        "--l2-coef",
        type=float,
        default=0.0,
        help="Additional L2 penalty coefficient on SAE features.",
    )
    parser.add_argument(
        "--pre-layer-norm",
        action="store_true",
        default=False,
        help="Apply non-affine LayerNorm to activations before the SAE encoder.",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument(
        "--k-warmup-fraction",
        type=float,
        default=0.2,
        help=(
            "For TopK/TopBottomK SAEs, start with all SAE features eligible and "
            "decay k to --top-k by this fraction of training."
        ),
    )
    parser.add_argument(
        "--k-warmup-decay-rate",
        type=float,
        default=5.0,
        help=(
            "Shape of TopK/TopBottomK k warmup decay. Larger values drop k faster "
            "early and slow more as k approaches --top-k."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--sae-dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-root", type=Path, default=PROJECT_ROOT / "custom_saes")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-every", type=int, default=200)
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
        help=(
            "Target average number of non-zero features per activation token. "
            "For ReLU/shrink SAEs this feedback adjusts the L1 coefficient."
        ),
    )
    parser.add_argument(
        "--sparsity-control-rate",
        type=float,
        default=0.005,
        help="Log-space feedback update rate for target sparsity control.",
    )
    parser.add_argument(
        "--sparsity-control-max-log-step",
        type=float,
        default=0.002,
        help="Maximum absolute log change applied to the L1 coefficient per training step.",
    )
    parser.add_argument(
        "--sparsity-control-deadband",
        type=float,
        default=0.05,
        help="Relative sparsity error tolerated before changing the L1 coefficient.",
    )
    parser.add_argument(
        "--sparsity-ema-beta",
        type=float,
        default=0.95,
        help="EMA smoothing factor for measured average non-zero features.",
    )
    parser.add_argument(
        "--min-l1-coef",
        type=float,
        default=1e-8,
        help="Minimum L1 coefficient used by target sparsity control.",
    )
    parser.add_argument(
        "--max-l1-coef",
        type=float,
        default=1e6,
        help="Maximum L1 coefficient used by target sparsity control.",
    )
    return parser.parse_args()


def resolve_run_name(run_name: Optional[str]) -> str:
    if run_name is None or not run_name.strip():
        return f"gemma3_270m_sae_{int(time.time())}"

    path = Path(run_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("--run-name must be a relative name under --save-root.")
    return path.as_posix()


def default_l1_for_activation(activation: str) -> float:
    return DEFAULT_L1_BY_ACTIVATION.get(activation.lower(), 1e-4)


def resolve_sae_variants(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.activations is None:
        return [dict(variant) for variant in SAE_VARIANTS]

    return [
        {
            "name": activation,
            "activation": activation,
            "l1_coef": default_l1_for_activation(activation),
        }
        for activation in args.activations
    ]


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


def layer_from_hook_point(hook_point: str) -> int:
    match = re.fullmatch(r"blocks\.(\d+)\.hook_[A-Za-z0-9_]+", hook_point)
    return int(match.group(1)) if match else -1


def resolve_training_hook_point(
    model,
    hook_layer: str,
    hook_site: str,
    hook_index: Optional[int] = None,
    hook_point: Optional[str] = None,
) -> tuple[str, int]:
    n_layers = int(model.cfg.n_layers)
    if hook_point is not None and hook_point.strip():
        resolved_hook_point = hook_point.strip()
        layer = layer_from_hook_point(resolved_hook_point)
        if layer >= n_layers:
            raise ValueError(
                f"--hook-point resolved to layer {layer}, but model has layers 0..{n_layers - 1}."
            )
        return resolved_hook_point, layer

    if hook_index is not None:
        layer = hook_index
    else:
        normalized_layer = hook_layer.strip().lower()

        if normalized_layer in ("middle", "mid"):
            layer = n_layers // 2
        elif normalized_layer in ("last", "end", "final"):
            layer = n_layers - 1
        else:
            try:
                layer = int(normalized_layer)
            except ValueError as exc:
                raise ValueError(
                    "--hook-layer must be 'middle', 'last'/'end', or a zero-indexed integer; "
                    f"got {hook_layer!r}."
                ) from exc

    if not 0 <= layer < n_layers:
        source = "--hook-index" if hook_index is not None else "--hook-layer"
        raise ValueError(f"{source} resolved to {layer}, but model has layers 0..{n_layers - 1}.")

    return HookPointSpec(layer=layer, site=hook_site).name, layer


def tokenize_text_batch(model, text_batch: list[str], device: str, context_size: int) -> ActivationBatch:
    encoded = model.tokenizer(
        text_batch,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=context_size,
        return_tensors="pt",
    )
    return ActivationBatch(
        tokens=encoded["input_ids"].to(device),
        attention_mask=encoded["attention_mask"].to(device).bool(),
    )


def collect_activations_for_text_batch(
    model,
    text_batch: list[str],
    hook_point: str,
    d_in: int,
    device: str,
    context_size: int,
    preserve_context: bool = False,
) -> ActivationBatch:
    token_batch = tokenize_text_batch(model, text_batch, device, context_size)
    if token_batch.tokens is None or token_batch.attention_mask is None:
        raise ValueError("Tokenization did not return tokens and an attention mask.")

    with torch.no_grad():
        _, cache = model.run_with_cache(
            token_batch.tokens,
            names_filter=hook_point,
            attention_mask=token_batch.attention_mask,
        )

    acts = cache[hook_point].detach().to(dtype=torch.float32)
    if preserve_context:
        return ActivationBatch(activations=acts.reshape(acts.shape[0], acts.shape[1], d_in), attention_mask=token_batch.attention_mask)
    return ActivationBatch(activations=acts[token_batch.attention_mask].reshape(-1, d_in))


def cat_padded_context_tensors(tensors: list[torch.Tensor], pad_value: float | bool | int = 0) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot concatenate an empty tensor list.")
    if len({tensor.shape[1] for tensor in tensors}) == 1:
        return torch.cat(tensors, dim=0)

    max_seq_len = max(tensor.shape[1] for tensor in tensors)
    padded_tensors: list[torch.Tensor] = []
    for tensor in tensors:
        if tensor.shape[1] == max_seq_len:
            padded_tensors.append(tensor)
            continue

        padded_shape = list(tensor.shape)
        padded_shape[1] = max_seq_len
        padded = tensor.new_full(padded_shape, pad_value)
        seq_slice = (slice(None), slice(0, tensor.shape[1]), *([slice(None)] * (tensor.ndim - 2)))
        padded[seq_slice] = tensor
        padded_tensors.append(padded)

    return torch.cat(padded_tensors, dim=0)


def activation_batches(
    model,
    texts: Iterable[str],
    hook_point: str,
    d_in: int,
    device: str,
    batch_tokens: int,
    model_forward_texts: int,
    context_size: int,
    preserve_context: bool = False,
) -> Iterator[ActivationBatch]:
    buffer: list[torch.Tensor] = []
    mask_buffer: list[torch.Tensor] = []
    buffered = 0

    for text_batch in batched(texts, model_forward_texts):
        collected = collect_activations_for_text_batch(
            model=model,
            text_batch=text_batch,
            hook_point=hook_point,
            d_in=d_in,
            device=device,
            context_size=context_size,
            preserve_context=preserve_context,
        )
        if collected.activations is None:
            raise ValueError("Activation collection did not return activations.")
        buffer.append(collected.activations)
        if preserve_context:
            if collected.attention_mask is None:
                raise ValueError("preserve_context=True requires an attention mask.")
            mask_buffer.append(collected.attention_mask)
        buffered += collected.token_count

        while buffered >= batch_tokens:
            if preserve_context:
                joined = cat_padded_context_tensors(buffer)
                joined_mask = cat_padded_context_tensors(mask_buffer, pad_value=False)
                cumulative_tokens = joined_mask.sum(dim=1).cumsum(dim=0)
                context_count = int((cumulative_tokens < batch_tokens).sum().item()) + 1
                yield ActivationBatch(activations=joined[:context_count], attention_mask=joined_mask[:context_count])
                remainder = joined[context_count:]
                remainder_mask = joined_mask[context_count:]
                buffer = [remainder] if remainder.numel() else []
                mask_buffer = [remainder_mask] if remainder_mask.numel() else []
                buffered = int(remainder_mask.sum().item()) if remainder_mask.numel() else 0
            else:
                joined = torch.cat(buffer, dim=0)
                yield ActivationBatch(activations=joined[:batch_tokens])
                remainder = joined[batch_tokens:]
                buffer = [remainder] if remainder.numel() else []
                buffered = remainder.shape[0] if remainder.numel() else 0

    if buffer:
        if preserve_context:
            yield ActivationBatch(
                activations=cat_padded_context_tensors(buffer),
                attention_mask=cat_padded_context_tensors(mask_buffer, pad_value=False),
            )
        else:
            yield ActivationBatch(activations=torch.cat(buffer, dim=0))


def token_batches(
    model,
    texts: Iterable[str],
    device: str,
    batch_tokens: int,
    model_forward_texts: int,
    context_size: int,
) -> Iterator[ActivationBatch]:
    buffer_tokens: list[torch.Tensor] = []
    buffer_masks: list[torch.Tensor] = []
    buffered = 0

    for text_batch in batched(texts, model_forward_texts):
        batch = tokenize_text_batch(model, text_batch, device, context_size)
        if batch.tokens is None or batch.attention_mask is None:
            raise ValueError("Tokenization did not return tokens and an attention mask.")
        buffer_tokens.append(batch.tokens)
        buffer_masks.append(batch.attention_mask)
        buffered += batch.token_count

        while buffered >= batch_tokens:
            joined_tokens = cat_padded_context_tensors(buffer_tokens)
            joined_mask = cat_padded_context_tensors(buffer_masks, pad_value=False)
            cumulative_tokens = joined_mask.sum(dim=1).cumsum(dim=0)
            context_count = int((cumulative_tokens < batch_tokens).sum().item()) + 1
            yield ActivationBatch(tokens=joined_tokens[:context_count], attention_mask=joined_mask[:context_count])
            remainder_tokens = joined_tokens[context_count:]
            remainder_mask = joined_mask[context_count:]
            buffer_tokens = [remainder_tokens] if remainder_tokens.numel() else []
            buffer_masks = [remainder_mask] if remainder_mask.numel() else []
            buffered = int(remainder_mask.sum().item()) if remainder_mask.numel() else 0

    if buffer_tokens:
        yield ActivationBatch(
            tokens=cat_padded_context_tensors(buffer_tokens),
            attention_mask=cat_padded_context_tensors(buffer_masks, pad_value=False),
        )


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


def set_sae_k(sae: TrainableSAE, k: int) -> None:
    k = max(1, min(int(k), sae.cfg.d_sae))
    sae.cfg.k = k
    if isinstance(sae.activation_fn, torch.nn.Module) and hasattr(sae.activation_fn, "k"):
        sae.activation_fn.k = k


def set_shrink_threshold(sae: TrainableSAE, threshold: float) -> None:
    sae.cfg.shrink_threshold = float(threshold)
    if isinstance(sae.activation_fn, torch.nn.Module) and hasattr(sae.activation_fn, "threshold"):
        sae.activation_fn.threshold = float(threshold)


def k_for_step(
    step: int,
    target_k: int,
    start_k: int,
    warmup_steps: int,
) -> int:
    if warmup_steps <= 1:
        return target_k
    progress = min(max((step - 1) / (warmup_steps - 1), 0.0), 1.0)
    return round(start_k + progress * (target_k - start_k))


def make_k_warmup_config(args: argparse.Namespace, sae: TrainableSAE, total_steps: int) -> Optional[dict[str, int]]:
    activation = sae.cfg.activation.lower()
    if not args.k_warmup or activation not in ("topk", "tbk", "topbottomk", "top_bottom_k"):
        return None
    if not 0 < args.k_warmup_start_frac <= 1:
        raise ValueError("--k-warmup-start-frac must be in (0, 1].")
    if not 0 < args.k_warmup_training_frac <= 1:
        raise ValueError("--k-warmup-training-frac must be in (0, 1].")

    target_k = max(1, min(args.top_k, sae.cfg.d_sae))
    start_k = max(target_k, min(sae.cfg.d_sae, round(sae.cfg.d_sae * args.k_warmup_start_frac)))
    warmup_steps = max(1, round(total_steps * args.k_warmup_training_frac))
    return {"target_k": target_k, "start_k": start_k, "warmup_steps": warmup_steps}


def make_shrink_threshold_warmup_config(
    sae: TrainableSAE,
    total_steps: int,
    training_frac: float = 0.20,
) -> Optional[dict[str, float]]:
    activation = sae.cfg.activation.lower()
    if activation not in ("shrink", "softshrink", "soft_shrink", "soft_threshold"):
        return None

    target_threshold = float(sae.cfg.shrink_threshold)
    warmup_steps = max(1, round(total_steps * training_frac))
    return {
        "target_threshold": target_threshold,
        "warmup_steps": float(warmup_steps),
    }


def shrink_threshold_for_step(step: int, target_threshold: float, warmup_steps: int) -> float:
    if warmup_steps <= 1:
        return target_threshold
    progress = min(max((step - 1) / (warmup_steps - 1), 0.0), 1.0)
    return progress * target_threshold


def build_sae(
    args: argparse.Namespace,
    variant: dict[str, object],
    d_in: int,
    hook_point: str,
    hook_layer: int,
    device: str,
    run_dir: Path,
):
    total_steps = args.max_steps or max(1, args.token_budget // args.batch_tokens)
    l1_coef = (
        float(args.l1_coef)
        if args.l1_coef is not None
        else float(variant["l1_coef"])
    )

    cfg = SAEConfig(
        d_in=d_in,
        d_sae=d_in * args.expansion_factor,
        activation=str(variant["activation"]),
        k=args.top_k,
        shrink_threshold=args.shrink_threshold,
        pre_layer_norm=args.pre_layer_norm,
        lr=args.lr,
        l1_coefficient=l1_coef,
        l1_context_coef=args.l1_context_coef,
        l2_coefficient=args.l2_coef,
        dtype=args.sae_dtype,
        device=device,
        metadata={
            "variant": variant["name"],
            "model_name": args.model,
            "hook_point": hook_point,
            "hook_layer": hook_layer,
            "hook_layer_arg": args.hook_layer,
            "hook_index_arg": args.hook_index,
            "hook_point_arg": args.hook_point,
            "hook_site": args.hook_site,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "token_budget": args.token_budget,
            "context_size": args.context_size,
            "model_forward_texts": args.model_forward_texts,
            "batch_tokens": args.batch_tokens,
            "loss_objective": args.loss_objective,
            "l1_coef": l1_coef,
            "l1_context_coef": args.l1_context_coef,
            "l2_coef": args.l2_coef,
            "k_warmup_fraction": args.k_warmup_fraction,
            "k_warmup_decay_rate": args.k_warmup_decay_rate,
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
    max_log_step: float,
    deadband: float,
    min_l1_coef: float,
    max_l1_coef: float,
) -> Optional[Callable[[TrainableSAE, dict[str, float]], None]]:
    if target_nonzero_features is None:
        return None
    if target_nonzero_features <= 0:
        raise ValueError("--target-nonzero-features must be positive.")
    if control_rate < 0:
        raise ValueError("--sparsity-control-rate must be non-negative.")
    if not 0 <= ema_beta < 1:
        raise ValueError("--sparsity-ema-beta must be in [0, 1).")
    if max_log_step < 0:
        raise ValueError("--sparsity-control-max-log-step must be non-negative.")
    if deadband < 0:
        raise ValueError("--sparsity-control-deadband must be non-negative.")
    if min_l1_coef < 0:
        raise ValueError("--min-l1-coef must be non-negative.")
    if max_l1_coef < min_l1_coef:
        raise ValueError("--max-l1-coef must be greater than or equal to --min-l1-coef.")

    ema_nonzero: Optional[float] = None

    def target_sparsity_udf(sae: TrainableSAE, metrics: dict[str, float]) -> None:
        nonlocal ema_nonzero
        avg_nonzero = metrics["avg_nonzero_features"]
        ema_nonzero = avg_nonzero if ema_nonzero is None else ema_beta * ema_nonzero + (1 - ema_beta) * avg_nonzero
        ratio = max(float(ema_nonzero) / target_nonzero_features, 1e-12)
        relative_error = ratio - 1.0
        log_error = math.log(ratio)
        deadband_log = math.log1p(deadband)
        controlled_log_error = 0.0
        if abs(log_error) > deadband_log:
            controlled_log_error = math.copysign(abs(log_error) - deadband_log, log_error)
        raw_log_update = control_rate * controlled_log_error
        log_update = max(-max_log_step, min(max_log_step, raw_log_update))
        multiplier = math.exp(log_update)

        activation = sae.cfg.activation.lower()
        is_l1_controlled_activation = activation == "relu" or activation in (
            "shrink",
            "softshrink",
            "soft_shrink",
            "soft_threshold",
        )
        if is_l1_controlled_activation:
            current_l1 = float(sae.cfg.l1_coefficient)
            if current_l1 <= 0.0 and controlled_log_error > 0.0:
                current_l1 = min_l1_coef
            new_l1 = current_l1 * multiplier
            sae.cfg.l1_coefficient = max(min_l1_coef, min(max_l1_coef, new_l1))
            metrics["l1_coef"] = float(sae.cfg.l1_coefficient)
            metrics["l1_coef_multiplier"] = float(multiplier)

        metrics["target_nonzero_features"] = float(target_nonzero_features)
        metrics["ema_nonzero_features"] = float(ema_nonzero)
        metrics["sparsity_error"] = float(relative_error)
        metrics["sparsity_ratio"] = float(ratio)
        metrics["sparsity_log_error"] = float(log_error)
        metrics["sparsity_log_update"] = float(log_update)

    return target_sparsity_udf


def train_cross_entropy_step(
    model,
    sae: TrainableSAE,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    hook_point: str,
    optimizer: torch.optim.Optimizer,
    clip_grad_norm: Optional[float] = 1.0,
    record_nonzero_features: bool = False,
    sparsity_udf: Optional[Callable[[TrainableSAE, dict[str, float]], None]] = None,
) -> dict[str, float]:
    sae.train()
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    captured_features: list[torch.Tensor] = []

    def sae_reconstruction_hook(acts: torch.Tensor, hook) -> torch.Tensor:
        del hook
        input_dtype = acts.dtype
        sae_dtype = next(sae.parameters()).dtype
        recon, features = sae(acts.to(device=sae.cfg.device, dtype=sae_dtype))
        captured_features.append(features)
        return recon.to(dtype=input_dtype)

    try:
        model.add_hook(hook_point, sae_reconstruction_hook)
        logits = model(tokens, attention_mask=attention_mask)
    finally:
        model.reset_hooks()

    if not captured_features:
        raise RuntimeError(f"SAE hook {hook_point!r} did not run during the model forward pass.")

    prediction_logits = logits[:, :-1, :].contiguous()
    target_tokens = tokens[:, 1:].contiguous()
    target_mask = attention_mask[:, 1:].to(dtype=prediction_logits.dtype)
    cross_entropy_per_token = F.cross_entropy(
        prediction_logits.view(-1, prediction_logits.shape[-1]),
        target_tokens.view(-1),
        reduction="none",
    ).view_as(target_tokens)
    cross_entropy = (cross_entropy_per_token * target_mask).sum() / target_mask.sum().clamp_min(1.0)

    features = captured_features[-1]
    regularization, reg_metrics = sae.regularization_loss(features, loss_mask=attention_mask)
    loss = cross_entropy + regularization
    metrics = {
        "loss": float(loss.detach()),
        "cross_entropy": float(cross_entropy.detach()),
        **reg_metrics,
    }

    if record_nonzero_features or sparsity_udf is not None:
        nonzero_features = (features != 0).sum(dim=-1).float()
        mask = attention_mask.to(device=features.device, dtype=nonzero_features.dtype)
        metrics["avg_nonzero_features"] = float(
            (nonzero_features * mask).sum().div(mask.sum().clamp_min(1.0)).detach()
        )

    loss.backward()
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(sae.parameters(), clip_grad_norm)
    optimizer.step()
    sae.normalize_decoder_weights()
    if sparsity_udf is not None:
        sparsity_udf(sae, metrics)
    return metrics




def main() -> None:
    args = parse_args()
    if not 0 <= args.k_warmup_fraction <= 1:
        raise ValueError("--k-warmup-fraction must be between 0 and 1.")
    if args.k_warmup_decay_rate <= 0:
        raise ValueError("--k-warmup-decay-rate must be positive.")
    if args.l1_coef is not None and args.l1_coef < 0:
        raise ValueError("--l1-coef must be non-negative.")
    if args.shrink_threshold < 0:
        raise ValueError("--shrink-threshold must be non-negative.")
    variants = resolve_sae_variants(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    model_dtype = getattr(torch, args.model_dtype)
    run_name = resolve_run_name(args.run_name)
    run_dir = args.save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print(f"Loading model {args.model} on {device} ({args.model_dtype})")
    model = load_hooked_transformer(args.model, device=device, dtype=model_dtype)
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    hook_point, hook_layer = resolve_training_hook_point(
        model,
        hook_layer=args.hook_layer,
        hook_site=args.hook_site,
        hook_index=args.hook_index,
        hook_point=args.hook_point,
    )
    d_in = model.cfg.d_model
    hook_layer_display = hook_layer if hook_layer >= 0 else "unknown"
    print(
        f"hook_point={hook_point}, hook_layer={hook_layer_display}, "
        f"hook_site={args.hook_site}, d_in={d_in}"
    )

    questions = read_questions(args.questions_path, args.question_repeats)

    print(f"Training variants sequentially: {', '.join(str(v['name']) for v in variants)}")
    print(f"Saving to {run_dir}")

    for variant in variants:
        name = str(variant["name"])
        print(f"\n=== Training {name} SAE ===")
        sae, optimizer, scheduler, variant_dir = build_sae(
            args=args,
            variant=variant,
            d_in=d_in,
            hook_point=hook_point,
            hook_layer=hook_layer,
            device=device,
            run_dir=run_dir,
        )
        metrics: list[dict[str, float]] = []
        texts = chain_questions_then_dataset(questions, args.dataset, args.dataset_config, args.split)
        preserve_context = args.l1_context_coef != 0.0
        total_seen_tokens = 0
        step = 0
        total_steps = args.max_steps or max(1, args.token_budget // args.batch_tokens)
        loss_since_update = 0.0
        steps_since_update = 0
        previous_update_avg_loss: Optional[float] = None
        update_checkpoint_dir = variant_dir / "best_step_update"
        sparsity_udf = make_target_sparsity_udf(
            args.target_nonzero_features,
            args.sparsity_control_rate,
            args.sparsity_ema_beta,
            args.sparsity_control_max_log_step,
            args.sparsity_control_deadband,
            args.min_l1_coef,
            args.max_l1_coef,
        )
        k_warmup = make_k_warmup_config(args, sae, total_steps)
        shrink_threshold_warmup = make_shrink_threshold_warmup_config(sae, total_steps)
        if k_warmup is not None:
            print(
                f"k warmup: {k_warmup['start_k']:,} -> {k_warmup['target_k']:,} "
                f"over {k_warmup['warmup_steps']:,} steps"
            )
        if shrink_threshold_warmup is not None:
            print(
                "shrink threshold warmup: "
                f"0.0000 -> {shrink_threshold_warmup['target_threshold']:.4f} "
                f"over {int(shrink_threshold_warmup['warmup_steps']):,} steps"
            )
            if sparsity_udf is not None:
                print(
                    "sparsity control starts after shrink threshold warmup "
                    f"at step {int(shrink_threshold_warmup['warmup_steps']) + 1:,}"
                )
        checkpoint_warmup_steps = max(
            min(args.warmup_steps, max(0, total_steps - 1)),
            k_warmup["warmup_steps"] if k_warmup is not None else 0,
            int(shrink_threshold_warmup["warmup_steps"]) if shrink_threshold_warmup is not None else 0,
        )
        if checkpoint_warmup_steps > 0:
            print(f"best_step_update saves start after warmup step {checkpoint_warmup_steps:,}")

        batches = (
            token_batches(
                model=model,
                texts=texts,
                device=device,
                batch_tokens=args.batch_tokens,
                model_forward_texts=args.model_forward_texts,
                context_size=args.context_size,
            )
            if args.loss_objective == "cross_entropy"
            else activation_batches(
                model=model,
                texts=texts,
                hook_point=hook_point,
                d_in=d_in,
                device=device,
                batch_tokens=args.batch_tokens,
                model_forward_texts=args.model_forward_texts,
                context_size=args.context_size,
                preserve_context=preserve_context,
            )
        )

        for step, batch in enumerate(
            batches,
            start=1,
        ):
            if k_warmup is not None:
                current_k = k_for_step(
                    step,
                    target_k=k_warmup["target_k"],
                    start_k=k_warmup["start_k"],
                    warmup_steps=k_warmup["warmup_steps"],
                )
                set_sae_k(sae, current_k)
            if shrink_threshold_warmup is not None:
                current_shrink_threshold = shrink_threshold_for_step(
                    step,
                    target_threshold=shrink_threshold_warmup["target_threshold"],
                    warmup_steps=int(shrink_threshold_warmup["warmup_steps"]),
                )
                set_shrink_threshold(sae, current_shrink_threshold)
            step_sparsity_udf = sparsity_udf
            if shrink_threshold_warmup is not None and step <= int(shrink_threshold_warmup["warmup_steps"]):
                step_sparsity_udf = None

            total_seen_tokens += batch.token_count

            if args.loss_objective == "cross_entropy":
                if batch.tokens is None or batch.attention_mask is None:
                    raise ValueError("Cross entropy training requires token batches with attention masks.")
                step_metrics = train_cross_entropy_step(
                    model=model,
                    sae=sae,
                    tokens=batch.tokens.to(device=device),
                    attention_mask=batch.attention_mask.to(device=device),
                    hook_point=hook_point,
                    optimizer=optimizer,
                    record_nonzero_features=args.log_nonzero_features,
                    sparsity_udf=step_sparsity_udf,
                )
            else:
                if batch.activations is None:
                    raise ValueError("Reconstruction training requires activation batches.")
                activations = batch.activations.to(device=device, dtype=torch.float32)
                attention_mask = (
                    batch.attention_mask.to(device=device)
                    if batch.attention_mask is not None
                    else None
                )
                step_metrics = sae.training_step(
                    activations,
                    optimizer,
                    record_nonzero_features=args.log_nonzero_features,
                    sparsity_udf=step_sparsity_udf,
                    loss_mask=attention_mask,
                )
            scheduler.step()
            step_metrics["lr"] = scheduler.get_last_lr()[0]
            if k_warmup is not None:
                step_metrics["k"] = float(sae.cfg.k)
            if shrink_threshold_warmup is not None:
                step_metrics["shrink_threshold"] = float(sae.cfg.shrink_threshold)
            metrics.append(step_metrics)
            loss_since_update += step_metrics["loss"]
            steps_since_update += 1

            if step == 1 or step % args.log_every == 0:
                update_avg_loss = loss_since_update / max(1, steps_since_update)
                improved_from_last_update = (
                    previous_update_avg_loss is None
                    or update_avg_loss < previous_update_avg_loss
                )
                step_metrics["update_avg_loss"] = float(update_avg_loss)
                if previous_update_avg_loss is not None:
                    step_metrics["previous_update_avg_loss"] = float(previous_update_avg_loss)
                checkpoint_text = ""
                checkpoint_warmup_complete = step >= checkpoint_warmup_steps
                step_metrics["checkpoint_warmup_complete"] = float(checkpoint_warmup_complete)
                if checkpoint_warmup_steps > 0:
                    step_metrics["checkpoint_warmup_steps"] = float(checkpoint_warmup_steps)
                if checkpoint_warmup_complete and improved_from_last_update:
                    step_metrics["saved_step_update_checkpoint"] = 1.0
                    save_one(update_checkpoint_dir, sae, metrics)
                    checkpoint_text = f" | saved={update_checkpoint_dir.name}"
                if checkpoint_warmup_complete:
                    previous_update_avg_loss = update_avg_loss
                loss_since_update = 0.0
                steps_since_update = 0

                nonzero_text = ""
                if args.log_nonzero_features:
                    nonzero_text = f" | avg_nonzero={step_metrics['avg_nonzero_features']:.2f}"
                if "l1_context" in step_metrics:
                    nonzero_text += f" | l1_context={step_metrics['l1_context']:.4f}"
                if "l2" in step_metrics:
                    nonzero_text += f" | l2={step_metrics['l2']:.4f}"
                if "k" in step_metrics:
                    nonzero_text += f" | k={int(step_metrics['k'])}"
                if "ema_nonzero_features" in step_metrics:
                    nonzero_text += f" | ema_nonzero={step_metrics['ema_nonzero_features']:.2f}"
                    if "l1_coef" in step_metrics:
                        nonzero_text += f" | l1_coef={step_metrics['l1_coef']:.2e}"
                if "shrink_threshold" in step_metrics:
                    nonzero_text += f" | shrink_threshold={step_metrics['shrink_threshold']:.4f}"
                print(
                    f"{name} | step {step:05d} | tokens={total_seen_tokens:,} | "
                    f"loss={step_metrics['loss']:.4f} | avg_loss={update_avg_loss:.4f} | {args.loss_objective}="
                    f"{step_metrics.get('mse', step_metrics.get('cross_entropy', 0.0)):.4f} | "
                    f"l1={step_metrics['l1']:.4f}{nonzero_text} | lr={step_metrics['lr']:.2e}"
                    f"{checkpoint_text}"
                )

            if args.max_steps is not None and step >= args.max_steps:
                break
            if total_seen_tokens >= args.token_budget:
                break

        if k_warmup is not None:
            set_sae_k(sae, k_warmup["target_k"])
        if shrink_threshold_warmup is not None:
            set_shrink_threshold(sae, shrink_threshold_warmup["target_threshold"])
        save_one(variant_dir, sae, metrics)
        print(f"Saved {name} SAE to {variant_dir}")
        print(f"Finished {step:,} steps over {total_seen_tokens:,} activation tokens for {name}.")
        free_sae_memory(sae, optimizer, scheduler, metrics)

    print(f"\nSaved all SAEs under {run_dir}")


if __name__ == "__main__":
    main()

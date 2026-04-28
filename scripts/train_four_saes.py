from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import random
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

from trainable_sae import HookPointSpec, SAEConfig, TrainableSAE, load_hooked_transformer, resolve_device


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
    # {"name": "relu", "activation": "relu", "l1_coefficient": 250.},
    # {"name": "shrink", "activation": "shrink", "l1_coefficient": 400},
    # {"name": "topk", "activation": "topk", "l1_coefficient": 0.0},
    {"name": "topbottomk", "activation": "tbk", "l1_coefficient": 10},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ReLU, TopK, TopBottomK, and shrink SAEs.")
    parser.add_argument("--model", default="google/gemma-3-270m-it")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--questions-path", type=Path, default=PROJECT_ROOT / "samples/questions_train.txt")
    parser.add_argument("--question-repeats", type=int, default=3)
    parser.add_argument("--token-budget", type=int, default=500_000_000)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--model-forward-texts", type=int, default=128)
    parser.add_argument("--batch-tokens", type=int, default=64)
    parser.add_argument(
        "--hook-layer",
        default="middle",
        help=(
            "Transformer block layer for SAE activations: 'middle', 'last'/'end', "
            "or an explicit zero-indexed layer number."
        ),
    )
    parser.add_argument(
        "--hook-site",
        default="resid_post",
        choices=("resid_pre", "resid_post", "mlp_out", "attn_out"),
        help="TransformerLens hook site within the selected block.",
    )
    parser.add_argument("--expansion-factor", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--shrink-threshold", type=float, default=.5)
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
        "--pre-layer-norm",
        action="store_true",
        default=False,
        help="Apply non-affine LayerNorm to activations before the SAE encoder.",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-7)
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


def resolve_training_hook_point(model, hook_layer: str, hook_site: str) -> tuple[str, int]:
    n_layers = int(model.cfg.n_layers)
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
        raise ValueError(f"--hook-layer resolved to {layer}, but model has layers 0..{n_layers - 1}.")

    return HookPointSpec(layer=layer, site=hook_site).name, layer


def tokenize_text_batch(model, text_batch: list[str], device: str, context_size: int) -> ActivationBatch:
    encoded = model.tokenizer(
        text_batch,
        add_special_tokens=True,
        padding=True,
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
                joined = torch.cat(buffer, dim=0)
                joined_mask = torch.cat(mask_buffer, dim=0)
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
            yield ActivationBatch(activations=torch.cat(buffer, dim=0), attention_mask=torch.cat(mask_buffer, dim=0))
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
            joined_tokens = torch.cat(buffer_tokens, dim=0)
            joined_mask = torch.cat(buffer_masks, dim=0)
            cumulative_tokens = joined_mask.sum(dim=1).cumsum(dim=0)
            context_count = int((cumulative_tokens < batch_tokens).sum().item()) + 1
            yield ActivationBatch(tokens=joined_tokens[:context_count], attention_mask=joined_mask[:context_count])
            remainder_tokens = joined_tokens[context_count:]
            remainder_mask = joined_mask[context_count:]
            buffer_tokens = [remainder_tokens] if remainder_tokens.numel() else []
            buffer_masks = [remainder_mask] if remainder_mask.numel() else []
            buffered = int(remainder_mask.sum().item()) if remainder_mask.numel() else 0

    if buffer_tokens:
        yield ActivationBatch(tokens=torch.cat(buffer_tokens, dim=0), attention_mask=torch.cat(buffer_masks, dim=0))


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
        l1_context_coef=args.l1_context_coef,
        dtype=args.sae_dtype,
        device=device,
        metadata={
            "variant": variant["name"],
            "model_name": args.model,
            "hook_point": hook_point,
            "hook_layer": args.hook_layer,
            "hook_site": args.hook_site,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "token_budget": args.token_budget,
            "context_size": args.context_size,
            "model_forward_texts": args.model_forward_texts,
            "batch_tokens": args.batch_tokens,
            "loss_objective": args.loss_objective,
            "l1_context_coef": args.l1_context_coef,
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    hook_point, hook_layer = resolve_training_hook_point(model, args.hook_layer, args.hook_site)
    d_in = model.cfg.d_model
    print(f"hook_point={hook_point}, hook_layer={hook_layer}, hook_site={args.hook_site}, d_in={d_in}")

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
        preserve_context = args.l1_context_coef != 0.0
        total_seen_tokens = 0
        step = 0
        sparsity_udf = make_target_sparsity_udf(
            args.target_nonzero_features,
            args.sparsity_control_rate,
            args.sparsity_ema_beta,
        )

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
                    sparsity_udf=sparsity_udf,
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
                    sparsity_udf=sparsity_udf,
                    loss_mask=attention_mask,
                )
            scheduler.step()
            step_metrics["lr"] = scheduler.get_last_lr()[0]
            metrics.append(step_metrics)

            if step == 1 or step % args.log_every == 0:
                nonzero_text = ""
                if args.log_nonzero_features:
                    nonzero_text = f" | avg_nonzero={step_metrics['avg_nonzero_features']:.2f}"
                if "l1_context" in step_metrics:
                    nonzero_text += f" | l1_context={step_metrics['l1_context']:.4f}"
                if sparsity_udf is not None:
                    nonzero_text += f" | ema_nonzero={step_metrics['ema_nonzero_features']:.2f}"
                    if "l1_coefficient" in step_metrics:
                        nonzero_text += f" | l1_coeff={step_metrics['l1_coefficient']:.2e}"
                    if "shrink_threshold" in step_metrics:
                        nonzero_text += f" | shrink_threshold={step_metrics['shrink_threshold']:.4f}"
                print(
                    f"{name} | step {step:05d} | tokens={total_seen_tokens:,} | "
                    f"loss={step_metrics['loss']:.4f} | {args.loss_objective}="
                    f"{step_metrics.get('mse', step_metrics.get('cross_entropy', 0.0)):.4f} | "
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

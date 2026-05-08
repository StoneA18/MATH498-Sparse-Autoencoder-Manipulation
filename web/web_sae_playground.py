"""
Browser playground for local TrainableSAE checkpoints.

Run:
    uv run python web/web_sae_playground.py --host 127.0.0.1 --port 8000

The server is intentionally dependency-light: it uses Python's stdlib HTTP server
and the repo's existing TransformerLens/TrainableSAE utilities. The model and SAE
are loaded lazily on the first analyze/generate request.
"""

from __future__ import annotations

import argparse
import csv as _csv
import datetime as _datetime
import json
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainable_sae import SAEConnector, TrainableSAE, load_hooked_transformer, resolve_device


DEFAULT_SAE_ROOTS = (PROJECT_ROOT / "saved_saes",)
DEFAULT_MODEL_NAME = "google/gemma-3-270m-it"


@dataclass(frozen=True)
class SAEOption:
    label: str
    path: Path
    model_name: str
    hook_point: str
    variant: str
    d_sae: int
    activation: str


@dataclass
class LoadedState:
    sae_path: Path
    model_name: str
    hook_point: str
    hook_point_label: str
    device: str
    model: Any
    sae: TrainableSAE
    connector: SAEConnector


def discover_trainable_saes(roots: tuple[Path, ...]) -> list[SAEOption]:
    options: list[SAEOption] = []
    for root in roots:
        if not root.exists():
            continue
        for checkpoint in sorted(root.glob("**/trainable_sae.pt")):
            sae_dir = checkpoint.parent
            cfg = load_sae_config_for_discovery(sae_dir)
            if cfg is None:
                continue

            metadata = cfg.get("metadata", {})
            model_name = str(first_scalar(metadata.get("model_name"), "google/gemma-3-270m-it"))
            hook_point = str(first_scalar(metadata.get("hook_point"), ""))
            if not hook_point:
                continue
            variant = str(first_scalar(metadata.get("variant"), sae_dir.name))
            activation = str(first_scalar(cfg.get("activation"), ""))
            d_sae = payload_int(cfg, "d_sae", 0)
            rel = sae_dir.relative_to(PROJECT_ROOT) if sae_dir.is_relative_to(PROJECT_ROOT) else sae_dir
            label = f"{rel} ({variant}, {activation or '?'}, {d_sae or '?'} features)"
            options.append(
                SAEOption(
                    label=label,
                    path=sae_dir,
                    model_name=model_name,
                    hook_point=hook_point,
                    variant=variant,
                    d_sae=d_sae,
                    activation=activation,
                )
            )
    return options


def load_sae_config_for_discovery(sae_dir: Path) -> Optional[dict[str, Any]]:
    """Load SAE config from either the legacy sidecar or the checkpoint payload."""
    for config_dir in (sae_dir, sae_dir.parent):
        config_path = config_dir / "config.json"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cfg = None
            if isinstance(cfg, dict):
                return cfg

    checkpoint_path = sae_dir / "trainable_sae.pt"
    if not checkpoint_path.exists():
        return None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return None
    cfg = checkpoint.get("cfg") if isinstance(checkpoint, dict) else None
    return cfg if isinstance(cfg, dict) else None


class PlaygroundRuntime:
    def __init__(
        self,
        sae_roots: tuple[Path, ...],
        device: str,
        model_dtype: str,
        local_files_only: bool,
    ) -> None:
        self.sae_roots = sae_roots
        self.device = resolve_device(device)
        self.model_dtype = model_dtype
        self.local_files_only = local_files_only
        self.options = discover_trainable_saes(sae_roots)
        self._state: Optional[LoadedState] = None
        self._lock = threading.Lock()

    def option_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "label": option.label,
                "path": str(option.path),
                "modelName": option.model_name,
                "hookPoint": option.hook_point,
                "variant": option.variant,
                "dSae": option.d_sae,
                "activation": option.activation,
            }
            for option in self.options
        ]

    def get_option(self, sae_path: str | None) -> SAEOption:
        if not self.options:
            roots = ", ".join(str(root) for root in self.sae_roots)
            raise ValueError(f"No TrainableSAE checkpoints found under: {roots}")
        if not sae_path:
            return self.options[0]
        requested = Path(sae_path).expanduser()
        for option in self.options:
            if option.path == requested or str(option.path) == sae_path:
                return option
        raise ValueError(f"Unknown SAE checkpoint: {sae_path}")

    def resolve_hook_point(
        self,
        option: SAEOption,
        hook_point_target: Any,
        hook_layer_index: Any = 0,
    ) -> tuple[str, str]:
        target = str(first_scalar(hook_point_target, "checkpoint"))
        if target in ("", "checkpoint", "trained"):
            return option.hook_point, "checkpoint hook"
        if target in ("beginning", "start", "input"):
            return "blocks.0.hook_resid_pre", "beginning of model"
        if target in ("layer", "index", "custom"):
            layer_index = int(first_scalar(hook_layer_index, 0))
            if layer_index < 0:
                raise ValueError("Hook layer index must be non-negative.")
            return f"blocks.{layer_index}.hook_resid_pre", f"layer {layer_index} resid_pre"
        raise ValueError(f"Unknown hook point target: {target}")

    def load(
        self,
        sae_path: str | None,
        hook_point_target: Any = "checkpoint",
        hook_layer_index: Any = 0,
    ) -> LoadedState:
        option = self.get_option(sae_path)
        hook_point, hook_point_label = self.resolve_hook_point(
            option,
            hook_point_target,
            hook_layer_index,
        )
        with self._lock:
            if (
                self._state is not None
                and self._state.sae_path == option.path
                and self._state.hook_point == hook_point
            ):
                return self._state
            if self._state is not None and self._state.sae_path == option.path:
                connector = SAEConnector(
                    model=self._state.model,
                    sae=self._state.sae,
                    hook_point=hook_point,
                    device=self.device,
                    preserve_error=True,
                )
                self._state = LoadedState(
                    sae_path=option.path,
                    model_name=option.model_name,
                    hook_point=hook_point,
                    hook_point_label=hook_point_label,
                    device=self.device,
                    model=self._state.model,
                    sae=self._state.sae,
                    connector=connector,
                )
                return self._state

            model = (
                self._state.model
                if self._state is not None and self._state.model_name == option.model_name
                else None
            )
            dtype = resolve_model_dtype(self.device, self.model_dtype)
            if model is None:
                model = load_hooked_transformer(
                    option.model_name,
                    device=self.device,
                    dtype=dtype,
                    local_files_only=self.local_files_only,
                )
                model.to(dtype)
                if (
                    getattr(model, "tokenizer", None) is not None
                    and model.tokenizer.pad_token_id is None
                ):
                    model.tokenizer.pad_token = model.tokenizer.eos_token

            sae = TrainableSAE.load(option.path, device=self.device)
            sae.eval()
            connector = SAEConnector(
                model=model,
                sae=sae,
                hook_point=hook_point,
                device=self.device,
                preserve_error=True,
            )
            self._state = LoadedState(
                sae_path=option.path,
                model_name=option.model_name,
                hook_point=hook_point,
                hook_point_label=hook_point_label,
                device=self.device,
                model=model,
                sae=sae,
                connector=connector,
            )
            return self._state


def resolve_model_dtype(device: str, model_dtype: str) -> torch.dtype:
    """Mirror the notebooks: bfloat16 on CUDA by default, float32 on CPU/MPS."""
    if not device.startswith("cuda"):
        return torch.float32
    return getattr(torch, model_dtype)


def notebook_default_device() -> str:
    return "cuda:2" if torch.cuda.is_available() else "cpu"


def parse_feature_ids(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, list):
        values: list[int] = []
        for item in raw:
            values.extend(parse_feature_ids(item))
        return values
    text = str(raw).replace(",", " ")
    return [int(part) for part in text.split() if part.strip()]


def first_scalar(value: Any, default: Any) -> Any:
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


def payload_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = first_scalar(payload.get(key), default)
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                value = first_scalar(json.loads(value), default)
            except json.JSONDecodeError:
                pass
    return int(value)


def payload_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = first_scalar(payload.get(key), default)
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                value = first_scalar(json.loads(value), default)
            except json.JSONDecodeError:
                pass
    return float(value)


def parse_float_values(raw: Any, default: float) -> list[float]:
    if raw is None:
        return [default]
    if isinstance(raw, list):
        values: list[float] = []
        for item in raw:
            values.extend(parse_float_values(item, default))
        return values or [default]
    if isinstance(raw, torch.Tensor):
        if raw.numel() == 0:
            return [default]
        return [float(value) for value in raw.detach().reshape(-1).cpu().tolist()]
    text = str(raw).strip()
    if not text:
        return [default]
    if text.startswith("[") and text.endswith("]"):
        try:
            return parse_float_values(json.loads(text), default)
        except json.JSONDecodeError:
            pass
    return [float(part) for part in text.replace(",", " ").split() if part.strip()] or [default]


def indexed_float(values: list[float], index: int, default: float) -> float:
    if not values:
        return default
    return values[index] if index < len(values) else values[-1]


def payload_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = first_scalar(payload.get(key), default)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def parse_token_selector(payload: dict[str, Any]) -> Any:
    raw = payload.get("tokenIndex", -1)
    if raw is None:
        return -1
    if isinstance(raw, (int, float)):
        return int(raw)
    text = str(raw).strip().lower()
    if not text:
        return -1
    if text == "all":
        return "all"
    parts = [part for part in text.replace(",", " ").split() if part]
    if len(parts) == 1:
        try:
            return int(parts[0])
        except ValueError:
            return -1
    indices: list[int] = []
    for part in parts:
        try:
            indices.append(int(part))
        except ValueError:
            continue
    if not indices:
        return -1
    return lambda _count, values=indices: values


def build_projector(config: dict[str, Any]) -> tuple[Optional[Callable[[torch.Tensor], torch.Tensor]], str]:
    projection = str(first_scalar(config.get("projection"), "identity"))
    feature_ids = parse_feature_ids(config.get("featureIds"))
    values = parse_float_values(config.get("value"), 0.0)
    factors = parse_float_values(config.get("factor"), 1.0)
    thresholds = parse_float_values(config.get("threshold"), 0.0)
    threshold = indexed_float(thresholds, 0, 0.0)
    top_k = max(1, payload_int(config, "topK", 50))

    if projection in ("identity", "none"):
        return None, "identity"

    def projector(features: torch.Tensor) -> torch.Tensor:
        out = features.clone()
        width = out.shape[-1]
        valid_ids = [
            (offset, idx)
            for offset, idx in enumerate(feature_ids)
            if 0 <= idx < width
        ]

        if projection == "clamp":
            for offset, idx in valid_ids:
                out[..., idx] = indexed_float(values, offset, 0.0)
        elif projection == "add":
            for offset, idx in valid_ids:
                out[..., idx] += indexed_float(values, offset, 0.0)
        elif projection == "scale":
            for offset, idx in valid_ids:
                out[..., idx] *= indexed_float(factors, offset, 1.0)
        elif projection == "zero":
            for _, idx in valid_ids:
                out[..., idx] = 0
        elif projection == "threshold":
            if valid_ids:
                for offset, idx in valid_ids:
                    feature_threshold = indexed_float(thresholds, offset, threshold)
                    out[..., idx] = torch.where(
                        out[..., idx].abs() >= feature_threshold,
                        out[..., idx],
                        torch.zeros_like(out[..., idx]),
                    )
            else:
                out = torch.where(out.abs() >= threshold, out, torch.zeros_like(out))
        elif projection == "top_abs_k":
            k = min(top_k, width)
            indices = out.abs().topk(k, dim=-1).indices
            kept = torch.zeros_like(out)
            out = kept.scatter(-1, indices, out.gather(-1, indices))
        else:
            raise ValueError(f"Unknown projection function: {projection}")
        return out

    return projector, projection


_PRESET_CACHE: dict[tuple[str, str], np.ndarray] = {}
_NOTEBOOK_VECTOR_CACHE: dict[tuple[str, str, str, int], torch.Tensor] = {}


def load_preset_vector(sae_path: Path, preset_type: str) -> np.ndarray:
    key = (str(sae_path), preset_type)
    if key in _PRESET_CACHE:
        return _PRESET_CACHE[key]

    experiment = ""
    if preset_type in ("happy", "sad"):
        experiment = "happy_sad"
    elif preset_type in ("hot", "cold"):
        experiment = "hot_cold"
    else:
        raise ValueError(f"Unknown preset type: {preset_type}")

    vector_path = sae_path / "experiments" / f"{experiment}_delta_avg.npy"
    if not vector_path.exists():
        raise ValueError(f"Missing preset vector: {vector_path}")

    vector = np.load(vector_path)
    _PRESET_CACHE[key] = vector
    return vector


def build_preset_projector(
    state: LoadedState,
    payload: dict[str, Any],
) -> tuple[Optional[Callable[[torch.Tensor], torch.Tensor]], str]:
    preset_type = str(first_scalar(payload.get("presetType"), "hot"))
    method = str(first_scalar(payload.get("presetMethod"), "add"))
    factor = payload_float(payload, "presetFactor", 10.0)
    top_k = payload_int(payload, "presetTopK", 1000)

    raw = load_preset_vector(state.sae_path, preset_type)
    if top_k > 0 and top_k < raw.size:
        flat = np.abs(raw).ravel()
        threshold = np.partition(flat, -top_k)[-top_k]
        raw = np.where(np.abs(raw) >= threshold, raw, 0.0)
    direction = -1.0 if preset_type in ("sad", "cold") else 1.0

    def projector(features: torch.Tensor) -> torch.Tensor:
        vector = torch.as_tensor(raw, device=features.device, dtype=features.dtype) * direction
        if method == "add":
            return features + factor * vector
        if method == "project":
            denom = torch.dot(vector, vector)
            if denom.abs() < 1e-8:
                return features
            dot = (features * vector).sum(dim=-1, keepdim=True)
            correction = factor * (dot / denom) * vector
            return torch.where(dot > 0, features, features - correction)
        raise ValueError(f"Unknown preset method: {method}")

    return projector, f"preset_{preset_type}_{method}"


def notebook_vector_defaults(experiment: str) -> tuple[str, str, str, int, float]:
    if experiment == "happy_sad":
        return "happy", "happy", "sad", 100, 6.0
    if experiment == "hot_cold":
        return "hot", "hot", "cold", 1000, 10.0
    raise ValueError(f"Unknown notebook experiment: {experiment}")


def notebook_delta_avg(state: LoadedState, experiment: str, top_k: int) -> torch.Tensor:
    cache_key = (str(state.sae_path), state.hook_point, experiment, int(top_k))
    if cache_key in _NOTEBOOK_VECTOR_CACHE:
        return _NOTEBOOK_VECTOR_CACHE[cache_key]

    preset_type, _, _, _, _ = notebook_vector_defaults(experiment)
    delta_avg = torch.as_tensor(
        load_preset_vector(state.sae_path, preset_type),
        dtype=torch.float32,
    ).clone()

    k = min(max(1, int(top_k)), delta_avg.numel())
    top_k_values, _ = torch.topk(torch.abs(delta_avg), k=k)
    min_top_k = top_k_values[-1]
    delta_avg[torch.abs(delta_avg) < min_top_k] = 0

    _NOTEBOOK_VECTOR_CACHE[cache_key] = delta_avg
    return delta_avg


def format_prompt_for_model(model: Any, prompt: str) -> str:
    model_name = str(
        getattr(getattr(model, "cfg", None), "model_name", "") or ""
    ).lower()
    if "gemma" in model_name and "it" in model_name:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    return prompt


def clean_generated_text(text: str) -> str:
    eot = text.find("<end_of_turn>")
    if eot != -1:
        text = text[:eot]
    return text.strip()


def token_strings_for_prompt(model: Any, prompt: str, device: str) -> list[str]:
    token_ids = model.to_tokens(prompt).to(device)[0].detach().cpu().tolist()
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        return [tokenizer.decode([int(token_id)]) for token_id in token_ids]
    return [str(int(token_id)) for token_id in token_ids]


def generation_kwargs(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": payload_int(payload, "maxNewTokens", 60),
        "temperature": payload_float(payload, "temperature", 0.8),
        "top_p": payload_float(payload, "topP", 0.95),
        "do_sample": payload_bool(payload, "doSample", False),
        "use_past_kv_cache": payload_bool(payload, "usePastKvCache", False),
        "verbose": False,
    }
    return kwargs


def reset_generation_seed(payload: dict[str, Any]) -> None:
    seed = payload_int(payload, "seed", 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_without_sae(state: LoadedState, prompt: str, payload: dict[str, Any]) -> str:
    tokens = state.model.to_tokens(prompt).to(state.device)
    with torch.no_grad():
        output = state.model.generate(tokens, **generation_kwargs(state.model, payload))
    new_tokens = output[0, tokens.shape[1] :]
    return clean_generated_text(state.model.to_string(new_tokens))


def generate_with_projection(state: LoadedState, prompt: str, payload: dict[str, Any]) -> str:
    steering_mode = str(first_scalar(payload.get("steeringMode"), "custom"))
    if steering_mode == "preset":
        projector, _ = build_preset_projector(state, payload)
    else:
        projector, _ = build_projector(payload)
    mode = "reconstruct" if payload_bool(payload, "saeEnabled", True) else "cache"
    projector_location = str(first_scalar(payload.get("projectorLocation"), "post_activation"))
    token_index = parse_token_selector(payload)

    if mode == "cache":
        return generate_without_sae(state, prompt, payload)

    with torch.no_grad():
        generated = state.connector.generate_with_sae(
            prompt,
            mode="reconstruct",
            sae_projector=projector,
            projector_token_index=token_index,
            projector_location=projector_location,
            clean=True,
            **generation_kwargs(state.model, payload),
        )
    return generated


def generate_notebook_delta_pair(state: LoadedState, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    experiment = str(first_scalar(payload.get("notebookExperiment"), "hot_cold"))
    _, positive_label, negative_label, default_top_k, default_factor = notebook_vector_defaults(experiment)
    method = str(first_scalar(payload.get("notebookMethod"), "add"))
    top_k = payload_int(payload, "notebookTopK", default_top_k)
    default_method_factor = 1.0 if method == "project" else default_factor
    factor = payload_float(payload, "notebookFactor", default_method_factor)
    token_index = "all"
    projector_location = "post_activation"
    delta_avg = notebook_delta_avg(state, experiment=experiment, top_k=top_k)
    steering_vector = delta_avg.to(device=state.device, dtype=next(state.sae.parameters()).dtype)
    generate_kwargs = generation_kwargs(state.model, payload)

    def add_positive_vector(features: torch.Tensor) -> torch.Tensor:
        return features + factor * steering_vector

    def add_negative_vector(features: torch.Tensor) -> torch.Tensor:
        return features - factor * steering_vector

    def project_positive_vector(features: torch.Tensor) -> torch.Tensor:
        vector = steering_vector
        denom = torch.dot(vector, vector)
        if denom.abs() < 1e-8:
            return features
        score = (features * vector).sum(dim=-1, keepdim=True)
        correction = factor * (score / denom) * vector
        return torch.where(score > 0, features, features - correction)

    def project_negative_vector(features: torch.Tensor) -> torch.Tensor:
        vector = -steering_vector
        denom = torch.dot(vector, vector)
        if denom.abs() < 1e-8:
            return features
        score = (features * vector).sum(dim=-1, keepdim=True)
        correction = factor * (score / denom) * vector
        return torch.where(score > 0, features, features - correction)

    if method == "project":
        positive_projector = project_positive_vector
        negative_projector = project_negative_vector
        steering_method = "vector_projection"
    elif method == "add":
        positive_projector = add_positive_vector
        negative_projector = add_negative_vector
        steering_method = "additive"
    else:
        raise ValueError(f"Unknown notebook steering method: {method}")

    tokens = state.model.to_tokens(prompt).to(state.device)
    prompt_len = tokens.shape[1]

    reset_generation_seed(payload)
    with torch.no_grad():
        baseline_tokens = state.model.generate(tokens, **generate_kwargs)
    baseline = state.model.to_string(baseline_tokens[0, prompt_len:]).strip()

    reset_generation_seed(payload)
    if not payload_bool(payload, "saeEnabled", True):
        positive_steered = baseline
        negative_steered = baseline
    else:
        with torch.no_grad():
            positive_steered = state.connector.generate_with_sae(
                prompt,
                mode="reconstruct",
                sae_projector=positive_projector,
                projector_token_index=token_index,
                projector_location=projector_location,
                **generate_kwargs,
            )

        with torch.no_grad():
            negative_steered = state.connector.generate_with_sae(
                prompt,
                mode="reconstruct",
                sae_projector=negative_projector,
                projector_token_index=token_index,
                projector_location=projector_location,
                **generate_kwargs,
            )

    return {
        "baseline": baseline,
        "positiveSteered": positive_steered,
        "negativeSteered": negative_steered,
        "hotSteered": positive_steered,
        "coldSteered": negative_steered,
        "positiveLabel": positive_label,
        "negativeLabel": negative_label,
        "notebookExperiment": experiment,
        "notebookVectorNonzero": int((delta_avg != 0).sum().item()),
        "notebookVectorTopK": int(top_k),
        "notebookFactor": float(factor),
        "notebookSteeringMethod": steering_method,
    }


def analyze_prompt(state: LoadedState, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    top_n = max(1, min(50, payload_int(payload, "topN", 12)))
    selection = str(first_scalar(payload.get("selection"), "topk"))
    tokens = token_strings_for_prompt(state.model, prompt, state.device)

    with torch.no_grad():
        features = state.connector.features_for_prompt(prompt, projector_location="post_activation")
        if features.ndim == 3:
            features = features[0]
        elif features.ndim > 3:
            features = features.reshape(-1, features.shape[-1])

    if selection == "tbk":
        indices = features.abs().topk(min(top_n, features.shape[-1]), dim=-1).indices
        values = features.gather(-1, indices)
        order = values.abs().argsort(dim=-1, descending=True)
        indices = indices.gather(-1, order)
        values = values.gather(-1, order)
    else:
        values, indices = features.topk(min(top_n, features.shape[-1]), dim=-1)

    rows = []
    featured_ids: set[int] = set()
    row_count = min(len(tokens), indices.shape[0])
    for pos, token in enumerate(tokens[:row_count]):
        row_features = [
            {
                "id": int(indices[pos, rank].detach().cpu().item()),
                "value": float(values[pos, rank].detach().cpu().item()),
            }
            for rank in range(indices.shape[-1])
        ]
        featured_ids.update(feature["id"] for feature in row_features)
        rows.append(
            {
                "position": pos,
                "token": token,
                "features": row_features,
            }
        )

    feature_activations = {
        str(feature_id): [
            float(value)
            for value in features[:row_count, feature_id].detach().cpu().tolist()
        ]
        for feature_id in sorted(featured_ids)
    }

    nonzero = int((features != 0).sum().item())
    return {
        "tokens": len(tokens),
        "featureWidth": int(features.shape[-1]),
        "hookPoint": state.hook_point,
        "hookPointLabel": state.hook_point_label,
        "nonzero": nonzero,
        "avgNonzeroPerToken": float((features != 0).sum(dim=-1).float().mean().item()),
        "featureActivations": feature_activations,
        "rows": rows,
    }

_LOG_CSV = Path(__file__).parent / "log.csv"
_RESPONSES_DIR = Path(__file__).parent / "saved_responses"
_STATIC_HTML = Path(__file__).parent / "static_sae_playground.html"
_EMBED_MARKER = '<script id="embedded-data" type="application/json">'
_CSV_HEADERS = [
    "timestamp", "prompt", "saePath", "saeEnabled",
    "maxNewTokens", "temperature", "topP", "seed",
    "steeringMode", "notebookExperiment", "notebookMethod",
    "notebookFactor", "notebookTopK", "projection",
    "projectorLocation", "hookPointTarget", "hookLayerIndex",
    "featureIds", "value", "factor", "threshold", "topK",
    "tokenIndex", "hookPoint", "hookPointLabel", "elapsedSeconds",
    "positiveLabel", "negativeLabel", "notebookVectorNonzero",
    "baselineFile", "positiveSteeredFile", "negativeSteeredFile",
]


def _update_static_html(record: dict) -> None:
    if not _STATIC_HTML.exists():
        return
    html = _STATIC_HTML.read_text(encoding="utf-8")
    start = html.find(_EMBED_MARKER)
    if start == -1:
        return
    data_start = start + len(_EMBED_MARKER)
    end = html.find("</script>", data_start)
    if end == -1:
        return
    try:
        records = json.loads(html[data_start:end].strip() or "[]")
    except (json.JSONDecodeError, ValueError):
        records = []
    records.append(record)
    new_html = html[:data_start] + json.dumps(records, ensure_ascii=False) + html[end:]
    _STATIC_HTML.write_text(new_html, encoding="utf-8")


def _log_generation(payload: dict, response: dict) -> None:
    _RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    ts = _datetime.datetime.now()
    slug = ts.strftime("%Y-%m-%d_%H-%M-%S-%f")

    baseline = response.get("baseline", "")
    positive = response.get("positiveSteered", response.get("hotSteered", ""))
    negative = response.get("negativeSteered", response.get("coldSteered", ""))

    def _save(name: str, text: str) -> str:
        filename = f"{slug}_{name}.txt"
        (_RESPONSES_DIR / filename).write_text(text, encoding="utf-8")
        return filename

    baseline_file = _save("baseline", baseline)
    positive_file = _save("positive", positive)
    negative_file = _save("negative", negative)

    # Append to CSV (filenames only — no response text)
    write_header = not _LOG_CSV.exists() or _LOG_CSV.stat().st_size == 0
    with _LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        if write_header:
            writer.writerow(_CSV_HEADERS)
        writer.writerow([
            ts.isoformat(),
            payload.get("prompt", ""),
            payload.get("saePath", ""),
            payload.get("saeEnabled", True),
            payload.get("maxNewTokens", 60),
            payload.get("temperature", 0.8),
            payload.get("topP", 0.95),
            payload.get("seed", 0),
            payload.get("steeringMode", ""),
            payload.get("notebookExperiment", ""),
            payload.get("notebookMethod", ""),
            payload.get("notebookFactor", ""),
            payload.get("notebookTopK", ""),
            payload.get("projection", ""),
            payload.get("projectorLocation", ""),
            payload.get("hookPointTarget", ""),
            payload.get("hookLayerIndex", ""),
            payload.get("featureIds", ""),
            payload.get("value", ""),
            payload.get("factor", ""),
            payload.get("threshold", ""),
            payload.get("topK", ""),
            payload.get("tokenIndex", ""),
            response.get("hookPoint", ""),
            response.get("hookPointLabel", ""),
            response.get("elapsedSeconds", ""),
            response.get("positiveLabel", ""),
            response.get("negativeLabel", ""),
            response.get("notebookVectorNonzero", ""),
            baseline_file,
            positive_file,
            negative_file,
        ])

    # Embed full record (with response text) into static HTML
    _update_static_html({
        "timestamp": ts.isoformat(),
        "prompt": payload.get("prompt", ""),
        "saePath": str(payload.get("saePath", "")),
        "steeringMode": payload.get("steeringMode", ""),
        "notebookExperiment": payload.get("notebookExperiment", ""),
        "notebookMethod": payload.get("notebookMethod", ""),
        "notebookFactor": payload.get("notebookFactor", ""),
        "notebookTopK": payload.get("notebookTopK", ""),
        "maxNewTokens": payload.get("maxNewTokens", ""),
        "temperature": payload.get("temperature", ""),
        "topP": payload.get("topP", ""),
        "hookPoint": response.get("hookPoint", ""),
        "hookPointLabel": response.get("hookPointLabel", ""),
        "elapsedSeconds": response.get("elapsedSeconds", ""),
        "positiveLabel": response.get("positiveLabel", ""),
        "negativeLabel": response.get("negativeLabel", ""),
        "notebookVectorNonzero": response.get("notebookVectorNonzero", ""),
        "baseline": baseline,
        "positiveSteered": positive,
        "negativeSteered": negative,
    })


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SAE Projection Playground</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17201b;
      --muted: #607066;
      --line: #d9e1dc;
      --panel: #ffffff;
      --page: #f4f7f5;
      --green: #1b7f5f;
      --green-dark: #0d5f47;
      --blue: #315d9f;
      --red: #a03a42;
      --shadow: 0 8px 22px rgba(27, 45, 35, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--page);
      color: var(--ink);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfb;
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 700;
    }
    .status {
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
    }
    main {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: calc(100vh - 54px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: #fbfcfb;
      padding: 16px;
      overflow: auto;
    }
    section.workspace {
      padding: 16px;
      display: grid;
      grid-template-rows: auto auto minmax(240px, 1fr);
      gap: 14px;
      overflow: auto;
    }
    .group {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      padding: 12px;
      margin-bottom: 12px;
    }
    .group h2 {
      margin: 0 0 10px;
      font-size: 13px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 650;
      margin: 10px 0 5px;
    }
    input, select, textarea, button {
      font: inherit;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid #c9d4ce;
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 8px 9px;
      font-size: 14px;
    }
    textarea {
      min-height: 132px;
      resize: vertical;
      line-height: 1.45;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .hidden {
      display: none !important;
    }
    .toggle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin: 8px 0;
      font-size: 14px;
      font-weight: 650;
    }
    .toggle input {
      width: 42px;
      height: 22px;
      accent-color: var(--green);
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    button {
      border: 1px solid transparent;
      border-radius: 6px;
      background: var(--green);
      color: #fff;
      padding: 8px 12px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--green-dark); }
    button.secondary {
      background: #fff;
      color: var(--ink);
      border-color: #c9d4ce;
    }
    button.secondary:hover { background: #edf3ef; }
    .outputs {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }
    .output {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: var(--shadow);
      min-height: 180px;
      overflow: hidden;
    }
    .output h2 {
      margin: 0;
      padding: 10px 12px;
      font-size: 14px;
      border-bottom: 1px solid var(--line);
      background: #f9fbfa;
    }
    pre {
      margin: 0;
      padding: 12px;
      white-space: pre-wrap;
      word-wrap: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      line-height: 1.45;
    }
    .analysis {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .analysis-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #f9fbfa;
      font-size: 13px;
      color: var(--muted);
    }
    .chart-panel {
      border-bottom: 1px solid var(--line);
      background: #fff;
      padding: 12px;
    }
    .chart-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .chart-title strong {
      color: var(--ink);
      font-size: 14px;
    }
    .chart-scroll {
      overflow-x: auto;
      border: 1px solid #edf1ee;
      border-radius: 6px;
      background: #fbfcfb;
    }
    #featureChart {
      display: block;
      min-width: 100%;
      height: 320px;
    }
    .chart-empty {
      color: var(--muted);
      font-size: 13px;
      padding: 22px 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid #edf1ee;
      padding: 7px 9px;
      vertical-align: top;
      text-align: left;
    }
    th {
      background: #fbfcfb;
      color: var(--muted);
      position: sticky;
      top: 0;
    }
    .token {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-weight: 700;
      white-space: pre-wrap;
    }
    .token-button {
      display: inline;
      border: 0;
      border-radius: 4px;
      background: transparent;
      color: inherit;
      padding: 2px 3px;
      font: inherit;
      text-align: left;
      white-space: pre-wrap;
      cursor: pointer;
    }
    .token-button:hover {
      background: #edf3ef;
      color: var(--green-dark);
    }
    .token-button.applied {
      background: #e6f1ed;
      color: var(--green-dark);
    }
    .pill {
      display: inline-flex;
      gap: 5px;
      align-items: baseline;
      border: 0;
      border-radius: 5px;
      background: #e6f1ed;
      color: #0c5d46;
      padding: 3px 6px;
      margin: 0 4px 4px 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      cursor: pointer;
    }
    .pill:hover { background: #d4e9e1; }
    .pill.active { background: var(--green); color: #fff; }
    .pill.active b { color: #fff; }
    .pill b { color: #17352b; }
    .error { color: var(--red); }
    .hint {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .outputs { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>SAE Projection Playground</h1>
    <div class="status" id="status">Starting...</div>
  </header>
  <main>
    <aside>
      <div class="group">
        <h2>Checkpoint</h2>
        <label for="saePath">SAE</label>
        <select id="saePath"></select>
        <div class="hint" id="saeMeta"></div>
      </div>

      <div class="group">
        <h2>Generation</h2>
        <div class="toggle">
          <span>SAE reconstruction</span>
          <input id="saeEnabled" type="checkbox" checked />
        </div>
        <div class="row">
          <div>
            <label for="maxNewTokens">New tokens</label>
            <input id="maxNewTokens" type="number" min="1" max="500" value="100" />
          </div>
          <div>
            <label for="temperature">Temperature</label>
            <input id="temperature" type="number" min="0" max="2" step="0.05" value="0.8" />
          </div>
        </div>
        <label for="topP">Top-p</label>
        <input id="topP" type="number" min="0.05" max="1" step="0.05" value="0.95" />
        <label for="seed">Seed</label>
        <input id="seed" type="number" value="42" disabled />
      </div>

      <div class="group">
        <h2>Projection</h2>
        <label for="steeringMode">Steering mode</label>
        <select id="steeringMode">
          <option value="notebook_delta_pair" selected>Saved delta pair</option>
          <option value="custom">Custom steering</option>
        </select>

        <div id="notebookControls">
          <label for="notebookExperiment">Experiment</label>
          <select id="notebookExperiment">
            <option value="hot_cold" selected>Hot / cold</option>
            <option value="happy_sad">Happy / sad</option>
          </select>
          <label for="notebookMethod">Method</label>
          <select id="notebookMethod">
            <option value="add" selected>Add vector</option>
            <option value="project">Vector projection</option>
          </select>
          <label for="notebookFactor" id="notebookFactorLabel">Factor</label>
          <input id="notebookFactor" inputmode="decimal" value="10" />
          <label for="notebookTopK">Top |k| features</label>
          <input id="notebookTopK" type="number" min="1" value="1000" />
          <div class="hint" id="notebookHint">Uses the saved hot_cold_delta_avg.npy for the selected SAE, then runs the notebook helper flow: baseline, +hot, and -cold.</div>
        </div>

        <div id="customControls">
        <label for="projection">Function</label>
        <select id="projection">
          <option value="identity">Identity</option>
          <option value="clamp">Clamp selected features</option>
          <option value="add">Add to selected features</option>
          <option value="scale">Scale selected features</option>
          <option value="zero">Zero selected features</option>
          <option value="threshold">Threshold by magnitude</option>
          <option value="top_abs_k">Keep top |k| features</option>
        </select>
        <label for="projectorLocation">Apply</label>
        <select id="projectorLocation">
          <option value="post_activation" selected>After SAE activation</option>
          <option value="pre_activation">Before SAE activation</option>
        </select>
        <label for="hookPointTarget">SAE hook point</label>
        <select id="hookPointTarget">
          <option value="checkpoint">Checkpoint hook</option>
          <option value="beginning">Beginning of model</option>
          <option value="layer">Layer index</option>
        </select>
        <label for="hookLayerIndex">Hook layer index</label>
        <input id="hookLayerIndex" type="number" min="0" step="1" value="0" />
        <label for="featureIds">Feature IDs</label>
        <input id="featureIds" placeholder="18493, 42, 9001" />
        <div class="row">
          <div>
            <label for="value">Value</label>
            <input id="value" inputmode="decimal" value="20" />
          </div>
          <div>
            <label for="factor">Factor</label>
            <input id="factor" inputmode="decimal" value="2" />
          </div>
        </div>
        <div class="row">
          <div>
            <label for="threshold">Threshold</label>
            <input id="threshold" inputmode="decimal" value="1" />
          </div>
          <div>
            <label for="topK">Top |k|</label>
            <input id="topK" type="number" min="1" value="50" />
          </div>
        </div>
        </div>
        <div id="tokenControls">
        <label for="tokenTarget">Target token</label>
        <select id="tokenTarget">
          <option value="newest">Newest generated token</option>
          <option value="beginning">Beginning of prompt</option>
          <option value="custom" selected>Custom index</option>
        </select>
        <label for="tokenIndex">Token index</label>
        <input id="tokenIndex" type="text" value="all" />
        <div class="hint">Use -1, 0, 5, a list like 1,2,3, or "all" for every token.</div>
        </div>
      </div>
    </aside>

    <section class="workspace">
      <div class="group">
        <label for="prompt">Prompt</label>
        <textarea id="prompt">is the temperature outside hot or cold?</textarea>
        <div class="actions" style="margin-top: 10px;">
          <button id="generate">Generate Compare</button>
          <button class="secondary" id="analyze">Analyze Features</button>
        </div>
      </div>

      <div class="outputs">
        <div class="output">
          <h2>Baseline</h2>
          <pre id="baselineOut">No generation yet.</pre>
        </div>
        <div class="output">
          <h2 id="positiveOutTitle">+ Hot Feature Vector</h2>
          <pre id="steeredOut">No generation yet.</pre>
        </div>
        <div class="output">
          <h2 id="negativeOutTitle">- Cold Feature Vector</h2>
          <pre id="coldOut">No generation yet.</pre>
        </div>
      </div>

      <div class="analysis">
        <div class="analysis-header">
          <span id="analysisMeta">Feature analysis appears here.</span>
          <span>
            <select id="selection">
              <option value="topk">Top positive</option>
              <option value="tbk">Top magnitude</option>
            </select>
            <input id="topN" type="number" min="1" max="50" value="12" style="width: 72px;" />
          </span>
        </div>
        <div class="chart-panel">
          <div class="chart-title">
            <strong id="chartFeature">Feature graph</strong>
            <span id="chartMeta">Click a feature pill, or Ctrl-click a token to load its features.</span>
          </div>
          <div class="chart-scroll" id="chartScroll">
            <div class="chart-empty" id="chartEmpty">Analyze a prompt, then click a feature to see activation by token.</div>
            <svg id="featureChart" role="img" aria-label="Feature activation by token"></svg>
          </div>
        </div>
        <div style="overflow:auto; max-height: 520px;">
          <table>
            <thead><tr><th style="width:70px;">Pos</th><th style="width:150px;">Token</th><th>Top Features</th></tr></thead>
            <tbody id="analysisRows"></tbody>
          </table>
        </div>
      </div>
    </section>
  </main>

<script>
const $ = (id) => document.getElementById(id);
let saeOptions = [];
let lastAnalysis = null;
let activeFeatureId = null;
let appliedTokenPosition = null;
const notebookConfigs = {
  hot_cold: {
    positive: "Hot",
    negative: "Cold",
    factor: "10",
    projectionFactor: "1",
    topK: "1000",
    vector: "hot_cold_delta_avg.npy",
    prompt: "is the temperature outside hot or cold?"
  },
  happy_sad: {
    positive: "Happy",
    negative: "Sad",
    factor: "6",
    projectionFactor: "1",
    topK: "100",
    vector: "happy_sad_delta_avg.npy",
    prompt: "tell me a story about a rainy afternoon"
  }
};

function payload() {
  const tokenTarget = $("tokenTarget").value;
  const tokenIndex = tokenTarget === "newest"
    ? "-1"
    : tokenTarget === "beginning"
      ? "0"
      : $("tokenIndex").value;
  return {
    saePath: $("saePath").value,
    prompt: $("prompt").value,
    saeEnabled: $("saeEnabled").checked,
    maxNewTokens: Number($("maxNewTokens").value),
    temperature: Number($("temperature").value),
    topP: Number($("topP").value),
    seed: Number($("seed").value),
    projection: $("projection").value,
    steeringMode: $("steeringMode").value,
    notebookExperiment: $("notebookExperiment").value,
    notebookMethod: $("notebookMethod").value,
    notebookFactor: $("notebookFactor").value,
    notebookTopK: Number($("notebookTopK").value),
    projectorLocation: $("projectorLocation").value,
    hookPointTarget: $("hookPointTarget").value,
    hookLayerIndex: Number($("hookLayerIndex").value),
    tokenIndex,
    featureIds: $("featureIds").value,
    value: $("value").value,
    factor: $("factor").value,
    threshold: $("threshold").value,
    topK: Number($("topK").value),
    selection: $("selection").value,
    topN: Number($("topN").value)
  };
}

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });
  const data = await res.json();
  if (!res.ok) {
    const err = new Error(data.error || `HTTP ${res.status}`);
    err.details = data.details || "";
    throw err;
  }
  return data;
}

function setBusy(text) {
  $("status").textContent = text;
}

function setError(target, err) {
  target.textContent = String(err.message || err);
  target.classList.add("error");
}

function clearError(target) {
  target.classList.remove("error");
}

async function loadStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  saeOptions = data.saes || [];
  $("saePath").innerHTML = saeOptions.map((s, i) =>
    `<option value="${s.path}">${s.label}</option>`
  ).join("");
  setDefaultSaeSelection();
  updateSaeMeta();
  $("status").textContent = saeOptions.length
    ? `Ready: ${saeOptions.length} SAE checkpoint(s), device ${data.device}`
    : "No local TrainableSAE checkpoints found.";
}

function updateSaeMeta() {
  const selected = saeOptions.find(s => s.path === $("saePath").value);
  $("saeMeta").textContent = selected
    ? `model ${selected.modelName} | hook ${selected.hookPoint}`
    : "";
}

function setDefaultSaeSelection() {
  const preferred = "saved_saes/shrink_mid_1/shrink/best_step_update";
  const match = saeOptions.find(s => s.path.endsWith(preferred));
  if (match) {
    $("saePath").value = match.path;
  }
}

function updateTokenIndexState() {
  const tokenTarget = $("tokenTarget").value;
  if (tokenTarget === "newest") {
    $("tokenIndex").value = "-1";
    $("tokenIndex").disabled = true;
  } else if (tokenTarget === "beginning") {
    $("tokenIndex").value = "0";
    $("tokenIndex").disabled = true;
  } else {
    $("tokenIndex").disabled = false;
  }
}

function updateHookLayerIndexState() {
  const usesLayerIndex = $("hookPointTarget").value === "layer";
  $("hookLayerIndex").disabled = !usesLayerIndex;
  if ($("hookPointTarget").value === "beginning") {
    $("hookLayerIndex").value = "0";
  }
}

function updateNotebookMethodState() {
  const experiment = $("notebookExperiment").value;
  const config = notebookConfigs[experiment] || notebookConfigs.hot_cold;
  const method = $("notebookMethod").value;
  const methodLabel = method === "project" ? "vector projection" : "additive vector";
  $("notebookFactorLabel").textContent = "Factor";
  $("notebookFactor").value = method === "project"
    ? config.projectionFactor
    : config.factor;
  $("notebookHint").textContent =
    `Uses the saved ${config.vector} for the selected SAE, then runs the ${methodLabel} notebook helper flow: baseline, +${config.positive.toLowerCase()}, and -${config.negative.toLowerCase()}.`;
}

function updateNotebookExperimentState() {
  const experiment = $("notebookExperiment").value;
  const config = notebookConfigs[experiment] || notebookConfigs.hot_cold;
  const defaultPrompts = new Set(Object.values(notebookConfigs).map(item => item.prompt));
  const currentPrompt = $("prompt").value.trim();

  $("positiveOutTitle").textContent = `+ ${config.positive} Feature Vector`;
  $("negativeOutTitle").textContent = `- ${config.negative} Feature Vector`;
  $("notebookTopK").value = config.topK;
  updateNotebookMethodState();

  if (!currentPrompt || defaultPrompts.has(currentPrompt)) {
    $("prompt").value = config.prompt;
  }
}

function updateSteeringModeState() {
  const mode = $("steeringMode").value;
  const useNotebook = mode === "notebook_delta_pair" || mode === "notebook_hot_cold";
  $("notebookControls").classList.toggle("hidden", !useNotebook);
  $("customControls").classList.toggle("hidden", useNotebook);
  $("tokenControls").classList.toggle("hidden", useNotebook);
  if (useNotebook) updateNotebookExperimentState();
}

async function generate() {
  clearError($("baselineOut"));
  clearError($("steeredOut"));
  clearError($("coldOut"));
  const experiment = $("notebookExperiment").value;
  const config = notebookConfigs[experiment] || notebookConfigs.hot_cold;
  $("baselineOut").textContent = "Generating baseline...";
  $("steeredOut").textContent = `Generating +${config.positive.toLowerCase()}...`;
  $("coldOut").textContent = `Generating -${config.negative.toLowerCase()}...`;
  setBusy("Loading model and generating...");
  try {
    const data = await api("/api/generate", payload());
    if (data.positiveLabel && data.negativeLabel) {
      $("positiveOutTitle").textContent = `+ ${capitalize(data.positiveLabel)} Feature Vector`;
      $("negativeOutTitle").textContent = `- ${capitalize(data.negativeLabel)} Feature Vector`;
    }
    $("baselineOut").textContent = data.baseline || "(empty)";
    $("steeredOut").textContent = data.positiveSteered || data.hotSteered || data.steered || "(empty)";
    $("coldOut").textContent = data.negativeSteered || data.coldSteered || "(empty)";
    const vectorText = data.notebookVectorNonzero === undefined
      ? ""
      : ` | saved vector nonzero ${data.notebookVectorNonzero}`;
    const methodText = data.notebookSteeringMethod === undefined
      ? ""
      : ` | method ${data.notebookSteeringMethod}`;
    setBusy(`Done in ${data.elapsedSeconds.toFixed(1)}s | ${data.hookPointLabel}: ${data.hookPoint}${vectorText}${methodText}`);
  } catch (err) {
    setError($("steeredOut"), err);
    setError($("coldOut"), err);
    setBusy("Generation failed.");
  }
}

async function analyze() {
  $("analysisRows").innerHTML = `<tr><td colspan="3">Analyzing...</td></tr>`;
  $("chartEmpty").style.display = "block";
  $("featureChart").innerHTML = "";
  activeFeatureId = null;
  appliedTokenPosition = null;
  setBusy("Loading model and analyzing features...");
  try {
    const data = await api("/api/analyze", payload());
    lastAnalysis = data;
    $("analysisMeta").textContent =
      `${data.tokens} tokens | ${data.featureWidth} features | ${data.hookPointLabel}: ${data.hookPoint} | avg nonzero ${data.avgNonzeroPerToken.toFixed(2)}`;
    $("analysisRows").innerHTML = data.rows.map(row => {
      const feats = row.features.map(f =>
        `<button class="pill" type="button" data-feature-id="${f.id}" onclick="handleFeaturePillClick(event, ${f.id}, ${f.value})"><b>${f.id}</b>${f.value.toFixed(3)}</button>`
      ).join("");
      return `<tr><td>${row.position}</td><td class="token"><button class="token-button" type="button" data-token-position="${row.position}" title="Click to graph the strongest feature. Ctrl/Cmd-click to load this token's top features.">${escapeHtml(row.token)}</button></td><td>${feats}</td></tr>`;
    }).join("");
    const firstFeature = data.rows?.[0]?.features?.[0]?.id;
    if (firstFeature !== undefined) renderFeatureChart(firstFeature);
    setBusy("Analysis complete.");
  } catch (err) {
    const detail = err.details ? `\n\n${err.details}` : "";
    $("analysisRows").innerHTML = `<tr><td colspan="3" class="error"><pre>${escapeHtml(String(err.message || err) + detail)}</pre></td></tr>`;
    setBusy("Analysis failed.");
  }
}

function splitNumberList(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return [];
  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) return parsed.map(Number).filter(Number.isFinite);
    if (Number.isFinite(Number(parsed))) return [Number(parsed)];
  } catch (_) {}
  return trimmed.split(/[,\s]+/).map(Number).filter(Number.isFinite);
}

function formatNumberList(values) {
  return values.map(v => Number(v).toFixed(6).replace(/\.?0+$/, "")).join(", ");
}

function addFeatureControlValue(featureId, featureValue) {
  const ids = splitNumberList($("featureIds").value).map(Number);
  const idIndex = ids.indexOf(Number(featureId));
  const targetIndex = idIndex === -1 ? ids.length : idIndex;
  const values = splitNumberList($("value").value);
  const fillValue = values.length ? values[values.length - 1] : Number(featureValue);

  while (values.length < ids.length) {
    values.push(fillValue);
  }
  if (idIndex === -1) {
    ids.push(Number(featureId));
    values.push(Number(featureValue));
  } else {
    values[targetIndex] = Number(featureValue);
  }

  $("featureIds").value = ids.join(", ");
  $("value").value = formatNumberList(values);
}

function setFeatureControlValues(features) {
  const cleanFeatures = (features || [])
    .map(f => ({id: Number(f.id), value: Number(f.value)}))
    .filter(f => Number.isInteger(f.id) && Number.isFinite(f.value));
  if (!cleanFeatures.length) return;

  $("featureIds").value = cleanFeatures.map(f => f.id).join(", ");
  $("value").value = formatNumberList(cleanFeatures.map(f => f.value));
  if ($("projection").value === "identity") {
    $("projection").value = "clamp";
  }
}

function handleFeaturePillClick(event, featureId, featureValue) {
  renderFeatureChart(featureId);
  if (event.ctrlKey || event.metaKey) {
    addFeatureControlValue(featureId, featureValue);
  }
}

function handleTokenClick(event, position) {
  if (!lastAnalysis) return;
  const row = (lastAnalysis.rows || []).find(r => Number(r.position) === Number(position));
  if (!row || !row.features || !row.features.length) return;

  renderFeatureChart(row.features[0].id);
  if (event.ctrlKey || event.metaKey) {
    setFeatureControlValues(row.features);
    appliedTokenPosition = Number(position);
    document.querySelectorAll(".token-button").forEach(el => {
      el.classList.toggle("applied", Number(el.dataset.tokenPosition) === appliedTokenPosition);
    });
    $("chartMeta").textContent = `loaded ${row.features.length} features from token ${position} "${compactToken(row.token)}"`;
    setBusy(`Loaded token ${position}'s feature values into projection controls.`);
  }
}

function renderFeatureChart(featureId) {
  if (!lastAnalysis || !lastAnalysis.featureActivations) return;
  const values = lastAnalysis.featureActivations[String(featureId)];
  if (!values) {
    $("chartFeature").textContent = `Feature ${featureId}`;
    $("chartMeta").textContent = "This feature was not included in the current top-feature set.";
    return;
  }

  activeFeatureId = featureId;
  document.querySelectorAll(".pill").forEach(el => {
    el.classList.toggle("active", Number(el.dataset.featureId) === featureId);
  });

  const rows = lastAnalysis.rows || [];
  const tokens = rows.map(row => row.token);
  const width = Math.max(760, values.length * 54 + 80);
  const height = 320;
  const pad = {left: 52, right: 24, top: 22, bottom: 76};
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const minV = Math.min(0, ...values);
  const maxV = Math.max(0, ...values);
  const span = Math.max(maxV - minV, 1e-9);
  const xStep = values.length > 1 ? plotW / (values.length - 1) : plotW;
  const y = (v) => pad.top + ((maxV - v) / span) * plotH;
  const zeroY = y(0);
  const barW = Math.max(8, Math.min(28, xStep * 0.56));
  const maxAbs = Math.max(...values.map(v => Math.abs(v)), 1e-9);
  const mean = values.reduce((acc, v) => acc + v, 0) / Math.max(values.length, 1);
  const peak = values.reduce((best, v, idx) =>
    Math.abs(v) > Math.abs(best.value) ? {value: v, idx} : best,
    {value: 0, idx: 0}
  );

  const bars = values.map((v, i) => {
    const x = pad.left + i * xStep - barW / 2;
    const top = Math.min(y(v), zeroY);
    const h = Math.max(1, Math.abs(y(v) - zeroY));
    const color = v < 0 ? "#a03a42" : "#1b7f5f";
    const token = tokens[i] ?? "";
    return `<rect x="${x.toFixed(2)}" y="${top.toFixed(2)}" width="${barW.toFixed(2)}" height="${h.toFixed(2)}" rx="3" fill="${color}" opacity="${(0.35 + 0.65 * Math.min(Math.abs(v) / maxAbs, 1)).toFixed(3)}">
      <title>${escapeHtml(token)} | ${v.toFixed(5)}</title>
    </rect>`;
  }).join("");

  const points = values.map((v, i) => `${(pad.left + i * xStep).toFixed(2)},${y(v).toFixed(2)}`).join(" ");
  const labels = values.map((v, i) => {
    const x = pad.left + i * xStep;
    const token = compactToken(tokens[i] ?? "");
    return `<text x="${x.toFixed(2)}" y="${height - 38}" text-anchor="end" transform="rotate(-45 ${x.toFixed(2)} ${height - 38})" font-size="11" fill="#607066">${escapeHtml(token)}</text>`;
  }).join("");
  const ticks = [minV, 0, maxV].map(v =>
    `<g><line x1="${pad.left}" x2="${width - pad.right}" y1="${y(v).toFixed(2)}" y2="${y(v).toFixed(2)}" stroke="#dfe7e2" />
       <text x="${pad.left - 10}" y="${(y(v) + 4).toFixed(2)}" text-anchor="end" font-size="11" fill="#607066">${v.toFixed(2)}</text></g>`
  ).join("");

  $("chartEmpty").style.display = "none";
  $("featureChart").setAttribute("viewBox", `0 0 ${width} ${height}`);
  $("featureChart").style.width = `${width}px`;
  $("featureChart").innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="#fbfcfb"></rect>
    ${ticks}
    <line x1="${pad.left}" x2="${width - pad.right}" y1="${zeroY.toFixed(2)}" y2="${zeroY.toFixed(2)}" stroke="#607066" stroke-width="1.4"></line>
    ${bars}
    <polyline points="${points}" fill="none" stroke="#315d9f" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"></polyline>
    ${points.split(" ").map((pt, i) => {
      const [cx, cy] = pt.split(",");
      return `<circle cx="${cx}" cy="${cy}" r="3.4" fill="#315d9f"><title>${escapeHtml(tokens[i] ?? "")} | ${values[i].toFixed(5)}</title></circle>`;
    }).join("")}
    ${labels}
  `;
  $("chartFeature").textContent = `Feature ${featureId}`;
  $("chartMeta").textContent = `peak ${peak.value.toFixed(3)} on "${compactToken(tokens[peak.idx] ?? "")}" | mean ${mean.toFixed(3)}`;
}

function compactToken(token) {
  const clean = String(token).replace(/\n/g, "\\n").replace(/\s/g, "·");
  return clean.length > 14 ? clean.slice(0, 13) + "…" : clean;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function capitalize(s) {
  const text = String(s || "");
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

$("generate").addEventListener("click", generate);
$("analyze").addEventListener("click", analyze);
$("saePath").addEventListener("change", updateSaeMeta);
$("hookPointTarget").addEventListener("change", updateHookLayerIndexState);
$("tokenTarget").addEventListener("change", updateTokenIndexState);
$("steeringMode").addEventListener("change", updateSteeringModeState);
$("notebookExperiment").addEventListener("change", updateNotebookExperimentState);
$("notebookMethod").addEventListener("change", updateNotebookMethodState);
$("analysisRows").addEventListener("click", event => {
  const tokenButton = event.target.closest(".token-button");
  if (tokenButton) {
    handleTokenClick(event, tokenButton.dataset.tokenPosition);
  }
});
loadStatus();
updateTokenIndexState();
updateHookLayerIndexState();
updateNotebookExperimentState();
updateSteeringModeState();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    runtime: PlaygroundRuntime

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(INDEX_HTML)
            return
        if parsed.path == "/api/status":
            self.send_json(
                {
                    "device": self.runtime.device,
                    "modelDtype": self.runtime.model_dtype,
                    "saes": self.runtime.option_payload(),
                }
            )
            return
        if parsed.path == "/archive":
            archive_path = Path(__file__).parent / "static_sae_playground.html"
            if archive_path.exists():
                self.send_html(archive_path.read_text(encoding="utf-8"))
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "static_sae_playground.html not found")
            return
        if parsed.path == "/log.csv":
            if _LOG_CSV.exists():
                data = _LOG_CSV.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "log.csv not found")
            return
        if parsed.path.startswith("/responses/"):
            filename = parsed.path[len("/responses/"):]
            if not filename or "/" in filename or "\\" in filename or ".." in filename:
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return
            file_path = _RESPONSES_DIR / filename
            if file_path.exists():
                data = file_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Response file not found")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self.read_json()
            if parsed.path == "/api/generate":
                self.handle_generate(payload)
                return
            if parsed.path == "/api/analyze":
                self.handle_analyze(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            details = traceback.format_exc(limit=8)
            print(details)
            self.send_json(
                {
                    "error": str(exc),
                    "details": details,
                },
                HTTPStatus.BAD_REQUEST,
            )

    def handle_generate(self, payload: dict[str, Any]) -> None:
        state = self.runtime.load(
            payload.get("saePath"),
            payload.get("hookPointTarget"),
            payload.get("hookLayerIndex"),
        )
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Prompt is empty.")
        formatted_prompt = format_prompt_for_model(state.model, prompt)
        start = time.perf_counter()
        steering_mode = str(first_scalar(payload.get("steeringMode"), "notebook_delta_pair"))
        if steering_mode in ("notebook_delta_pair", "notebook_hot_cold"):
            response = generate_notebook_delta_pair(state, formatted_prompt, payload)
        else:
            reset_generation_seed(payload)
            baseline = generate_without_sae(state, formatted_prompt, payload)
            reset_generation_seed(payload)
            steered = generate_with_projection(state, formatted_prompt, payload)
            response = {
                "baseline": baseline,
                "hotSteered": steered,
                "coldSteered": "",
            }
        response.update(
            {
                "hookPoint": state.hook_point,
                "hookPointLabel": state.hook_point_label,
                "elapsedSeconds": time.perf_counter() - start,
            }
        )
        try:
            _log_generation(payload, response)
        except Exception:
            pass
        self.send_json(response)

    def handle_analyze(self, payload: dict[str, Any]) -> None:
        state = self.runtime.load(
            payload.get("saePath"),
            payload.get("hookPointTarget"),
            payload.get("hookLayerIndex"),
        )
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Prompt is empty.")
        formatted_prompt = format_prompt_for_model(state.model, prompt)
        self.send_json(analyze_prompt(state, formatted_prompt, payload))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local SAE projection playground.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device",
        default=notebook_default_device(),
        help="Torch device. Defaults to the notebook pattern: cuda:2 if CUDA is available, else cpu.",
    )
    parser.add_argument("--model-dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use already-downloaded Hugging Face model/tokenizer files.",
    )
    parser.add_argument(
        "--sae-root",
        action="append",
        type=Path,
        help="Directory to scan for trainable_sae.pt checkpoints. Can be passed more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = tuple(args.sae_root) if args.sae_root else DEFAULT_SAE_ROOTS
    Handler.runtime = PlaygroundRuntime(
        sae_roots=roots,
        device=args.device,
        model_dtype=args.model_dtype,
        local_files_only=args.local_files_only,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving SAE playground at http://{args.host}:{args.port}")
    print("Model and SAE load lazily on the first analyze/generate request.")
    server.serve_forever()


if __name__ == "__main__":
    main()

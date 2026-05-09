"""
Programmatic Gemma + TrainableSAE generation helpers.

This module exposes the website's saved-vector generation path as a Python API.
It intentionally focuses on the preset delta vectors used by the playground:
hot/cold and happy/sad, with additive or projection steering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import threading
import time
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch

from trainable_sae import SAEConnector, TrainableSAE, load_hooked_transformer, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SAE_ROOTS = (PROJECT_ROOT / "saved_saes",)
DEFAULT_MODEL_NAME = "google/gemma-3-270m-it"

ExperimentName = Literal["hot_cold", "happy_sad"]
SteeringMethod = Literal["add", "project"]
PresetName = Literal["hot", "cold", "happy", "sad"]
ResponseDirection = Literal["positive", "negative", "both"]


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


@dataclass(frozen=True)
class PresetGenerationConfig:
    sae_path: str | Path | None = None
    experiment: ExperimentName = "hot_cold"
    preset: PresetName | None = None
    method: SteeringMethod = "add"
    factor: float | None = None
    top_k: int | None = None
    direction: ResponseDirection = "both"
    sae_enabled: bool = True
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int = 42
    do_sample: bool = False
    use_past_kv_cache: bool = False
    hook_point_target: str = "checkpoint"
    hook_layer_index: int = 0


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


def payload_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = first_scalar(payload.get(key), default)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


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
            model_name = str(first_scalar(metadata.get("model_name"), DEFAULT_MODEL_NAME))
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


def best_step_update_options(options: list[SAEOption]) -> list[SAEOption]:
    return [option for option in options if option.path.name == "best_step_update"]


def resolve_model_dtype(device: str, model_dtype: str) -> torch.dtype:
    """Mirror the notebooks: bfloat16 on CUDA by default, float32 on CPU/MPS."""
    if not device.startswith("cuda"):
        return torch.float32
    return getattr(torch, model_dtype)


def notebook_default_device() -> str:
    return "cuda:2" if torch.cuda.is_available() else "cpu"


class PlaygroundRuntime:
    def __init__(
        self,
        sae_roots: tuple[Path, ...] = DEFAULT_SAE_ROOTS,
        device: str | None = None,
        model_dtype: str = "bfloat16",
        local_files_only: bool = False,
        best_step_only: bool = False,
    ) -> None:
        self.sae_roots = sae_roots
        self.device = resolve_device(device or notebook_default_device())
        self.model_dtype = model_dtype
        self.local_files_only = local_files_only
        options = discover_trainable_saes(sae_roots)
        self.options = best_step_update_options(options) if best_step_only else options
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

    def get_option(self, sae_path: str | Path | None) -> SAEOption:
        if not self.options:
            roots = ", ".join(str(root) for root in self.sae_roots)
            raise ValueError(f"No TrainableSAE checkpoints found under: {roots}")
        if not sae_path:
            return self.options[0]
        requested = Path(sae_path).expanduser()
        for option in self.options:
            if option.path == requested or str(option.path) == str(sae_path):
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
        sae_path: str | Path | None,
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


_PRESET_CACHE: dict[tuple[str, str], np.ndarray] = {}
_NOTEBOOK_VECTOR_CACHE: dict[tuple[str, str, str, int], torch.Tensor] = {}
_GENERATOR_CACHE: dict[tuple[tuple[str, ...], str, str, bool], "GemmaSaeGenerator"] = {}


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
    return text.strip()


def token_strings_for_prompt(model: Any, prompt: str, device: str) -> list[str]:
    token_ids = model.to_tokens(prompt).to(device)[0].detach().cpu().tolist()
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        return [tokenizer.decode([int(token_id)]) for token_id in token_ids]
    return [str(int(token_id)) for token_id in token_ids]


def generation_kwargs(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    del model
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


def generate_notebook_delta_pair(
    state: LoadedState,
    prompt: str,
    payload: dict[str, Any],
    baseline: str | None = None,
) -> dict[str, Any]:
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

    if baseline is None:
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


def _infer_experiment_and_direction(
    experiment: ExperimentName,
    preset: PresetName | None,
    direction: ResponseDirection,
) -> tuple[ExperimentName, ResponseDirection]:
    if preset is None:
        return experiment, direction
    if preset == "hot":
        return "hot_cold", "positive"
    if preset == "cold":
        return "hot_cold", "negative"
    if preset == "happy":
        return "happy_sad", "positive"
    if preset == "sad":
        return "happy_sad", "negative"
    raise ValueError(f"Unknown preset: {preset}")


def preset_payload(config: PresetGenerationConfig) -> dict[str, Any]:
    experiment, _ = _infer_experiment_and_direction(
        config.experiment,
        config.preset,
        config.direction,
    )
    payload: dict[str, Any] = {
        "saePath": str(config.sae_path) if config.sae_path is not None else None,
        "saeEnabled": config.sae_enabled,
        "maxNewTokens": config.max_new_tokens,
        "temperature": config.temperature,
        "topP": config.top_p,
        "seed": config.seed,
        "doSample": config.do_sample,
        "usePastKvCache": config.use_past_kv_cache,
        "steeringMode": "notebook_delta_pair",
        "notebookExperiment": experiment,
        "notebookMethod": config.method,
        "hookPointTarget": config.hook_point_target,
        "hookLayerIndex": config.hook_layer_index,
    }
    if config.factor is not None:
        payload["notebookFactor"] = config.factor
    if config.top_k is not None:
        payload["notebookTopK"] = config.top_k
    return payload


def _selected_response(
    payload: dict[str, Any],
    response: dict[str, Any],
    config: PresetGenerationConfig,
) -> str | dict[str, str]:
    _, direction = _infer_experiment_and_direction(
        payload["notebookExperiment"],
        config.preset,
        config.direction,
    )
    positive_label = str(response.get("positiveLabel", "positive"))
    negative_label = str(response.get("negativeLabel", "negative"))
    if direction == "positive":
        return str(response["positiveSteered"])
    if direction == "negative":
        return str(response["negativeSteered"])
    return {
        positive_label: str(response["positiveSteered"]),
        negative_label: str(response["negativeSteered"]),
    }


class GemmaSaeGenerator:
    """Reusable runtime for Gemma 270M-IT plus local TrainableSAE steering."""

    def __init__(
        self,
        sae_roots: tuple[Path, ...] = DEFAULT_SAE_ROOTS,
        device: str | None = None,
        model_dtype: str = "bfloat16",
        local_files_only: bool = False,
    ) -> None:
        self.runtime = PlaygroundRuntime(
            sae_roots=sae_roots,
            device=device,
            model_dtype=model_dtype,
            local_files_only=local_files_only,
        )

    def generate(
        self,
        prompt: str,
        config: PresetGenerationConfig | None = None,
        baseline: str | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        if config is None:
            config = PresetGenerationConfig(**overrides)
        elif overrides:
            config = PresetGenerationConfig(**{**config.__dict__, **overrides})

        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("Prompt is empty.")

        payload = preset_payload(config)
        state = self.runtime.load(
            config.sae_path,
            config.hook_point_target,
            config.hook_layer_index,
        )
        formatted_prompt = format_prompt_for_model(state.model, prompt)

        start = time.perf_counter()
        response = generate_notebook_delta_pair(
            state,
            formatted_prompt,
            payload,
            baseline=baseline,
        )
        response.update(
            {
                "response": _selected_response(payload, response, config),
                "hookPoint": state.hook_point,
                "hookPointLabel": state.hook_point_label,
                "elapsedSeconds": time.perf_counter() - start,
            }
        )
        return response

    def generate_baseline(
        self,
        prompt: str,
        config: PresetGenerationConfig | None = None,
        **overrides: Any,
    ) -> str:
        if config is None:
            config = PresetGenerationConfig(**overrides)
        elif overrides:
            config = PresetGenerationConfig(**{**config.__dict__, **overrides})

        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("Prompt is empty.")

        payload = preset_payload(config)
        state = self.runtime.load(
            config.sae_path,
            config.hook_point_target,
            config.hook_layer_index,
        )
        formatted_prompt = format_prompt_for_model(state.model, prompt)
        reset_generation_seed(payload)
        return generate_without_sae(state, formatted_prompt, payload)


def generate_gemma_sae_response(
    prompt: str,
    *,
    sae_path: str | Path | None = None,
    experiment: ExperimentName = "hot_cold",
    preset: PresetName | None = None,
    method: SteeringMethod = "add",
    factor: float | None = None,
    top_k: int | None = None,
    direction: ResponseDirection = "both",
    sae_enabled: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 42,
    do_sample: bool = False,
    use_past_kv_cache: bool = False,
    hook_point_target: str = "checkpoint",
    hook_layer_index: int = 0,
    device: str | None = None,
    model_dtype: str = "bfloat16",
    local_files_only: bool = False,
    sae_roots: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    """Generate with the same saved-vector steering path used by the website."""
    roots = sae_roots or DEFAULT_SAE_ROOTS
    resolved_device = resolve_device(device or notebook_default_device())
    cache_key = (tuple(str(root) for root in roots), resolved_device, model_dtype, local_files_only)
    generator = _GENERATOR_CACHE.get(cache_key)
    if generator is None:
        generator = GemmaSaeGenerator(
            sae_roots=roots,
            device=resolved_device,
            model_dtype=model_dtype,
            local_files_only=local_files_only,
        )
        _GENERATOR_CACHE[cache_key] = generator

    config = PresetGenerationConfig(
        sae_path=sae_path,
        experiment=experiment,
        preset=preset,
        method=method,
        factor=factor,
        top_k=top_k,
        direction=direction,
        sae_enabled=sae_enabled,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        do_sample=do_sample,
        use_past_kv_cache=use_past_kv_cache,
        hook_point_target=hook_point_target,
        hook_layer_index=hook_layer_index,
    )
    return generator.generate(prompt, config)

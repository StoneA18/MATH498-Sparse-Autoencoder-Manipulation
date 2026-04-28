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

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainable_sae import SAEConnector, TrainableSAE, load_hooked_transformer, resolve_device


DEFAULT_SAE_ROOTS = (PROJECT_ROOT / "custom_saes", PROJECT_ROOT / "saved_saes")


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
            config_path = sae_dir / "config.json"
            if not config_path.exists():
                continue
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
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

    def load(self, sae_path: str | None) -> LoadedState:
        option = self.get_option(sae_path)
        with self._lock:
            if self._state is not None and self._state.sae_path == option.path:
                return self._state

            dtype = getattr(torch, self.model_dtype)
            model = load_hooked_transformer(
                option.model_name,
                device=self.device,
                dtype=dtype,
                local_files_only=self.local_files_only,
            )
            if getattr(model, "tokenizer", None) is not None and model.tokenizer.pad_token_id is None:
                model.tokenizer.pad_token = model.tokenizer.eos_token

            sae = TrainableSAE.load(option.path, device=self.device)
            sae.eval()
            connector = SAEConnector(
                model=model,
                sae=sae,
                hook_point=option.hook_point,
                device=self.device,
                preserve_error=True,
            )
            self._state = LoadedState(
                sae_path=option.path,
                model_name=option.model_name,
                hook_point=option.hook_point,
                device=self.device,
                model=model,
                sae=sae,
                connector=connector,
            )
            return self._state


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


def payload_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = first_scalar(payload.get(key), default)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def build_projector(config: dict[str, Any]) -> tuple[Optional[Callable[[torch.Tensor], torch.Tensor]], str]:
    projection = str(first_scalar(config.get("projection"), "identity"))
    feature_ids = parse_feature_ids(config.get("featureIds"))
    value = payload_float(config, "value", 0.0)
    factor = payload_float(config, "factor", 1.0)
    threshold = payload_float(config, "threshold", 0.0)
    top_k = max(1, payload_int(config, "topK", 50))

    if projection in ("identity", "none"):
        return None, "identity"

    def projector(features: torch.Tensor) -> torch.Tensor:
        out = features.clone()
        width = out.shape[-1]
        valid_ids = [idx for idx in feature_ids if 0 <= idx < width]

        if projection == "clamp":
            for idx in valid_ids:
                out[..., idx] = value
        elif projection == "add":
            for idx in valid_ids:
                out[..., idx] += value
        elif projection == "scale":
            for idx in valid_ids:
                out[..., idx] *= factor
        elif projection == "zero":
            for idx in valid_ids:
                out[..., idx] = 0
        elif projection == "threshold":
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


def format_prompt(model: Any, prompt: str) -> str:
    model_name = str(getattr(model.cfg, "model_name", "") or "").lower()
    if "gemma" in model_name and "it" in model_name:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    return prompt


def clean_generated_text(text: str) -> str:
    for marker in ("<end_of_turn>", "<eos>"):
        if marker in text:
            text = text.split(marker)[0]
    return text.replace("<bos>", "").strip()


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
        "verbose": False,
    }
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if eot_id is not None and eot_id != tokenizer.unk_token_id:
            kwargs["eos_token_id"] = eot_id
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
    projector, projection_name = build_projector(payload)
    mode = "reconstruct" if payload_bool(payload, "saeEnabled", True) else "cache"
    projector_location = str(first_scalar(payload.get("projectorLocation"), "post_activation"))
    token_index = payload_int(payload, "tokenIndex", -1)

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
    if projection_name != "identity":
        return clean_generated_text(generated)
    return generated


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
        "nonzero": nonzero,
        "avgNonzeroPerToken": float((features != 0).sum(dim=-1).float().mean().item()),
        "featureActivations": feature_activations,
        "rows": rows,
    }


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
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
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
      grid-template-columns: 1fr 1fr;
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
            <input id="maxNewTokens" type="number" min="1" max="500" value="60" />
          </div>
          <div>
            <label for="temperature">Temperature</label>
            <input id="temperature" type="number" min="0" max="2" step="0.05" value="0.8" />
          </div>
        </div>
        <label for="topP">Top-p</label>
        <input id="topP" type="number" min="0.05" max="1" step="0.05" value="0.95" />
        <label for="seed">Seed</label>
        <input id="seed" type="number" value="0" />
      </div>

      <div class="group">
        <h2>Projection</h2>
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
          <option value="post_activation">After SAE activation</option>
          <option value="pre_activation">Before SAE activation</option>
        </select>
        <label for="tokenIndex">Token index</label>
        <input id="tokenIndex" type="number" value="-1" />
        <div class="hint">Use -1 for the newest token during generation. Other values target a fixed position in the current forward pass.</div>
        <label for="featureIds">Feature IDs</label>
        <input id="featureIds" placeholder="18493, 42, 9001" />
        <div class="row">
          <div>
            <label for="value">Value</label>
            <input id="value" type="number" step="0.1" value="20" />
          </div>
          <div>
            <label for="factor">Factor</label>
            <input id="factor" type="number" step="0.1" value="2" />
          </div>
        </div>
        <div class="row">
          <div>
            <label for="threshold">Threshold</label>
            <input id="threshold" type="number" step="0.1" value="1" />
          </div>
          <div>
            <label for="topK">Top |k|</label>
            <input id="topK" type="number" min="1" value="50" />
          </div>
        </div>
      </div>
    </aside>

    <section class="workspace">
      <div class="group">
        <label for="prompt">Prompt</label>
        <textarea id="prompt">Tell me a concise story about a lighthouse that remembers every storm.</textarea>
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
          <h2>SAE / Projection</h2>
          <pre id="steeredOut">No generation yet.</pre>
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
            <span id="chartMeta">Click any feature pill below.</span>
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

function payload() {
  return {
    saePath: $("saePath").value,
    prompt: $("prompt").value,
    saeEnabled: $("saeEnabled").checked,
    maxNewTokens: Number($("maxNewTokens").value),
    temperature: Number($("temperature").value),
    topP: Number($("topP").value),
    seed: Number($("seed").value),
    projection: $("projection").value,
    projectorLocation: $("projectorLocation").value,
    tokenIndex: Number($("tokenIndex").value),
    featureIds: $("featureIds").value,
    value: Number($("value").value),
    factor: Number($("factor").value),
    threshold: Number($("threshold").value),
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

async function generate() {
  clearError($("baselineOut"));
  clearError($("steeredOut"));
  $("baselineOut").textContent = "Generating baseline...";
  $("steeredOut").textContent = "Generating SAE/projection...";
  setBusy("Loading model and generating...");
  try {
    const data = await api("/api/generate", payload());
    $("baselineOut").textContent = data.baseline || "(empty)";
    $("steeredOut").textContent = data.steered || "(empty)";
    setBusy(`Done in ${data.elapsedSeconds.toFixed(1)}s`);
  } catch (err) {
    setError($("steeredOut"), err);
    setBusy("Generation failed.");
  }
}

async function analyze() {
  $("analysisRows").innerHTML = `<tr><td colspan="3">Analyzing...</td></tr>`;
  $("chartEmpty").style.display = "block";
  $("featureChart").innerHTML = "";
  activeFeatureId = null;
  setBusy("Loading model and analyzing features...");
  try {
    const data = await api("/api/analyze", payload());
    lastAnalysis = data;
    $("analysisMeta").textContent =
      `${data.tokens} tokens | ${data.featureWidth} features | avg nonzero ${data.avgNonzeroPerToken.toFixed(2)}`;
    $("analysisRows").innerHTML = data.rows.map(row => {
      const feats = row.features.map(f =>
        `<button class="pill" type="button" data-feature-id="${f.id}" onclick="renderFeatureChart(${f.id})"><b>${f.id}</b>${f.value.toFixed(3)}</button>`
      ).join("");
      return `<tr><td>${row.position}</td><td class="token">${escapeHtml(row.token)}</td><td>${feats}</td></tr>`;
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

$("generate").addEventListener("click", generate);
$("analyze").addEventListener("click", analyze);
$("saePath").addEventListener("change", updateSaeMeta);
loadStatus();
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
        state = self.runtime.load(payload.get("saePath"))
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Prompt is empty.")
        formatted_prompt = format_prompt(state.model, prompt)
        start = time.perf_counter()
        reset_generation_seed(payload)
        baseline = generate_without_sae(state, formatted_prompt, payload)
        reset_generation_seed(payload)
        steered = generate_with_projection(state, formatted_prompt, payload)
        self.send_json(
            {
                "baseline": baseline,
                "steered": steered,
                "elapsedSeconds": time.perf_counter() - start,
            }
        )

    def handle_analyze(self, payload: dict[str, Any]) -> None:
        state = self.runtime.load(payload.get("saePath"))
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Prompt is empty.")
        formatted_prompt = format_prompt(state.model, prompt)
        self.send_json(analyze_prompt(state, formatted_prompt, payload))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local SAE projection playground.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-dtype", default="float32", choices=("float32", "bfloat16", "float16"))
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

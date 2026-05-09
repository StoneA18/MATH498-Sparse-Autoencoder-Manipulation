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
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sae_generation as sae_gen


DEFAULT_SAE_ROOTS = sae_gen.DEFAULT_SAE_ROOTS


def build_projector(config: dict[str, Any]) -> tuple[Optional[Callable[[torch.Tensor], torch.Tensor]], str]:
    projection = str(sae_gen.first_scalar(config.get("projection"), "identity"))
    feature_ids = sae_gen.parse_feature_ids(config.get("featureIds"))
    values = sae_gen.parse_float_values(config.get("value"), 0.0)
    factors = sae_gen.parse_float_values(config.get("factor"), 1.0)
    thresholds = sae_gen.parse_float_values(config.get("threshold"), 0.0)
    threshold = sae_gen.indexed_float(thresholds, 0, 0.0)
    top_k = max(1, sae_gen.payload_int(config, "topK", 50))

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
                out[..., idx] = sae_gen.indexed_float(values, offset, 0.0)
        elif projection == "add":
            for offset, idx in valid_ids:
                out[..., idx] += sae_gen.indexed_float(values, offset, 0.0)
        elif projection == "scale":
            for offset, idx in valid_ids:
                out[..., idx] *= sae_gen.indexed_float(factors, offset, 1.0)
        elif projection == "zero":
            for _, idx in valid_ids:
                out[..., idx] = 0
        elif projection == "threshold":
            if valid_ids:
                for offset, idx in valid_ids:
                    feature_threshold = sae_gen.indexed_float(thresholds, offset, threshold)
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


def generate_with_projection(state: sae_gen.LoadedState, prompt: str, payload: dict[str, Any]) -> str:
    steering_mode = str(sae_gen.first_scalar(payload.get("steeringMode"), "custom"))
    if steering_mode == "preset":
        projector, _ = sae_gen.build_preset_projector(state, payload)
    else:
        projector, _ = build_projector(payload)
    mode = "reconstruct" if sae_gen.payload_bool(payload, "saeEnabled", True) else "cache"
    projector_location = str(sae_gen.first_scalar(payload.get("projectorLocation"), "post_activation"))
    token_index = sae_gen.parse_token_selector(payload)

    if mode == "cache":
        return sae_gen.generate_without_sae(state, prompt, payload)

    with torch.no_grad():
        generated = state.connector.generate_with_sae(
            prompt,
            mode="reconstruct",
            sae_projector=projector,
            projector_token_index=token_index,
            projector_location=projector_location,
            clean=True,
            **sae_gen.generation_kwargs(state.model, payload),
        )
    return generated


def analyze_prompt(state: sae_gen.LoadedState, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    top_n = max(1, min(50, sae_gen.payload_int(payload, "topN", 12)))
    selection = str(sae_gen.first_scalar(payload.get("selection"), "topk"))
    tokens = sae_gen.token_strings_for_prompt(state.model, prompt, state.device)

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
    runtime: sae_gen.PlaygroundRuntime

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
        formatted_prompt = sae_gen.format_prompt_for_model(state.model, prompt)
        start = time.perf_counter()
        steering_mode = str(sae_gen.first_scalar(payload.get("steeringMode"), "notebook_delta_pair"))
        if steering_mode in ("notebook_delta_pair", "notebook_hot_cold"):
            response = sae_gen.generate_notebook_delta_pair(state, formatted_prompt, payload)
        else:
            sae_gen.reset_generation_seed(payload)
            baseline = sae_gen.generate_without_sae(state, formatted_prompt, payload)
            sae_gen.reset_generation_seed(payload)
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
        formatted_prompt = sae_gen.format_prompt_for_model(state.model, prompt)
        self.send_json(analyze_prompt(state, formatted_prompt, payload))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local SAE projection playground.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device",
        default=sae_gen.notebook_default_device(),
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
    Handler.runtime = sae_gen.PlaygroundRuntime(
        sae_roots=roots,
        device=args.device,
        model_dtype=args.model_dtype,
        local_files_only=args.local_files_only,
        best_step_only=True,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving SAE playground at http://{args.host}:{args.port}")
    print("Model and SAE load lazily on the first analyze/generate request.")
    server.serve_forever()


if __name__ == "__main__":
    main()

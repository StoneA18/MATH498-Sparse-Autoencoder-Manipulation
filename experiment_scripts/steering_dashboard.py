"""
Steering Dashboard experiment: compare multiple SAE feature-steering methods
in a single interactive HTML output.

Typical usage:

    from steering_utils import (
        load_gpt2_small, load_sae_from_neuronpedia, Dist,
        ClampOp, CondDistOp, EveryOtherTokenOp, FibonacciTokensOp,
    )
    from experiment_scripts.steering_dashboard import SteeringExperiment

    model = load_gpt2_small()
    sae   = load_sae_from_neuronpedia("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    model.add_sae(sae)

    exp = SteeringExperiment(model, feature_ids=[18493])
    exp.add_method("baseline",    {})
    exp.add_method("clamp_40",    {18493: ClampOp(40)})
    exp.add_method("cond_dist",   {18493: CondDistOp(0.3, Dist("normal", 40, 10))})
    exp.add_method("every_other", {18493: EveryOtherTokenOp(40)})

    results = exp.run("I like sharks", n_tokens=100)
    results.save_html("output/dashboard.html")
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import torch

from steering_utils import SteerableModel, SteerableSAE, SteeringOp

_ARCHIVE_DIR = Path("experiment_scripts/experiment_data/dashboard_experiment")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ExperimentResults:
    """Holds per-method generation results; renders to a self-contained HTML dashboard.

    Feature log format: each entry is ``{feature_id: {"v": float, "n": float}}``
    where ``v`` is the post-intervention activation and ``n`` is the natural
    (pre-intervention) activation. Equal values mean the op did not fire.
    """

    def __init__(
        self,
        prompt: str,
        feature_ids: list[int],
        n_prompt_tokens: int,
        model_name: str = "",
        hook_name: str = "",
    ) -> None:
        self.prompt = prompt
        self.feature_ids = feature_ids
        self.n_prompt_tokens = n_prompt_tokens
        self.model_name = model_name
        self.hook_name = hook_name
        self._methods: list[str] = []
        self._descriptions: dict[str, str] = {}
        self._results: dict[str, dict] = {}

    def add_method(
        self,
        name: str,
        tokens: list[str],
        feature_log: list[dict],
        description: str = "",
    ) -> None:
        self._methods.append(name)
        self._descriptions[name] = description
        self._results[name] = {"tokens": tokens, "feature_log": feature_log}

    def save_html(self, path: str) -> None:
        """Render the dashboard, write to *path*, and archive a timestamped copy."""
        html = _build_dashboard_html(self)
        primary = Path(path)
        primary.parent.mkdir(parents=True, exist_ok=True)
        primary.write_text(html, encoding="utf-8")
        print(f"Dashboard saved  → {primary}")

        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archive = _ARCHIVE_DIR / f"dashboard_{stamp}.html"
        _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archive.write_text(html, encoding="utf-8")
        print(f"Archive copy     → {archive}")

    def _to_json(self) -> dict:
        return {
            "prompt": self.prompt,
            "n_prompt_tokens": self.n_prompt_tokens,
            "feature_ids": self.feature_ids,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "methods": self._methods,
            "method_descriptions": self._descriptions,
            "results": self._results,
        }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class SteeringExperiment:
    """
    Registers named steering methods and runs them all against a single prompt,
    producing an ExperimentResults you can render to HTML.

    Args:
        model:       A SteerableModel with at least one SAE attached.
        feature_ids: Feature IDs to track across all methods (shown in tooltips).
        sae_idx:     Which attached SAE to apply ops to (default 0).
    """

    def __init__(
        self,
        model: SteerableModel,
        feature_ids: list[int],
        sae_idx: int = 0,
    ) -> None:
        self.model = model
        self.feature_ids = feature_ids
        self.sae_idx = sae_idx
        self._methods: list[tuple[str, dict]] = []

    def add_method(
        self,
        name: str,
        ops: dict,  # {feature_id: SteeringOp}
    ) -> "SteeringExperiment":
        """Register a method. Pass an empty dict for unsteered baseline. Returns self."""
        self._methods.append((name, ops))
        return self

    def run(self, prompt: str, n_tokens: int = 100) -> ExperimentResults:
        """Run every registered method and return the collected results."""
        if not self.model._saes:
            raise RuntimeError("No SAEs attached to model. Call model.add_sae() first.")

        sae = self.model._saes[self.sae_idx]
        input_ids = self.model._model.to_tokens(prompt, prepend_bos=True)
        n_prompt_tokens = int(input_ids.shape[1])
        model_name = getattr(self.model._model.cfg, "model_name", "")

        results = ExperimentResults(
            prompt=prompt,
            feature_ids=self.feature_ids,
            n_prompt_tokens=n_prompt_tokens,
            model_name=model_name,
            hook_name=sae.hook_name,
        )

        for method_name, ops in self._methods:
            print(f"  [{method_name}] generating {n_tokens} tokens…")
            tokens, feature_log = self._run_one(sae, input_ids, ops, n_tokens)
            desc = ", ".join(f"{fid}: {op!r}" for fid, op in ops.items()) or "(no ops — baseline)"
            results.add_method(method_name, tokens, feature_log, description=desc)

        return results

    # ------------------------------------------------------------------

    def _run_one(
        self,
        sae: SteerableSAE,
        input_ids: torch.Tensor,
        ops: dict,
        n_tokens: int,
    ) -> tuple[list[str], list[dict]]:
        hook_fn, log = sae._make_tracking_hook(ops, self.feature_ids)
        self.model._model.add_hook(sae.hook_name, hook_fn, dir="fwd")

        try:
            output = self.model._model.generate(
                input_ids,
                max_new_tokens=n_tokens,
                verbose=False,
                use_past_kv_cache=True,
            )
        finally:
            self.model._model.reset_hooks()

        if isinstance(output, str):
            all_tokens = self.model._model.to_str_tokens(output)
        else:
            all_tokens = self.model._model.to_str_tokens(output[0])

        return list(all_tokens), log


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _build_dashboard_html(results: ExperimentResults) -> str:
    data_json = json.dumps(results._to_json(), ensure_ascii=False)
    return _DASHBOARD_TEMPLATE.replace("__DATA_JSON__", data_json)


_DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SAE Steering Dashboard</title>
<style>
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: system-ui, -apple-system, sans-serif;
  margin: 0; padding: 16px 24px;
  background: #f5f6f8; color: #1a1a2e;
}
h1 { margin: 0 0 3px; font-size: 1.1rem; font-weight: 600; }
.meta { color: #6c757d; font-size: 0.8rem; margin-bottom: 14px; font-family: monospace; }

/* Controls bar */
.controls {
  display: flex; gap: 14px; align-items: center; flex-wrap: wrap;
  background: #fff; padding: 10px 16px;
  border: 1px solid #dee2e6; border-radius: 6px;
  margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.controls label { font-size: 0.86rem; font-weight: 500; white-space: nowrap; }
.controls select {
  padding: 4px 8px; border: 1px solid #ced4da; border-radius: 4px;
  font-size: 0.86rem; cursor: pointer; max-width: 240px;
}
.controls select:focus { outline: 2px solid #4361ee; border-color: transparent; }
.method-desc { font-size: 0.76rem; color: #6c757d; font-family: monospace; flex: 1; min-width: 0; }

/* Token section */
.card {
  background: #fff; border: 1px solid #dee2e6; border-radius: 6px;
  padding: 14px 16px; margin-bottom: 12px;
  box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.card-title {
  font-size: 0.72rem; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase; color: #6c757d; margin: 0 0 10px;
}
.legend { display: flex; gap: 14px; margin-bottom: 10px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 0.76rem; color: #555; }
.swatch { width: 13px; height: 13px; border-radius: 2px; border: 1px solid rgba(0,0,0,.1); }
#token-display { display: flex; flex-wrap: wrap; gap: 3px; line-height: 1; }
.tok {
  display: inline-block; padding: 4px 6px; border-radius: 3px;
  font-family: monospace; font-size: 0.84rem; cursor: default;
  border: 1.5px solid transparent; white-space: pre;
  transition: border-color .1s, transform .1s;
}
.tok:hover { border-color: #1a1a2e !important; transform: translateY(-1px); z-index: 5; }
.tok.is-bos  { background: #e9ecef !important; color: #6c757d !important; font-style: italic; font-size: 0.72rem; }
.tok.is-prompt { border-color: #93b4ef; }

/* Tooltip */
#tt {
  display: none; position: fixed; z-index: 9999; pointer-events: none;
  background: #1a1a2e; color: #e9ecef; padding: 8px 12px;
  border-radius: 6px; font-size: 0.78rem; max-width: 280px;
  box-shadow: 0 4px 14px rgba(0,0,0,.25);
}
#tt .tt-head { font-size: 0.7rem; color: #94b8f0; margin-bottom: 5px; padding-bottom: 4px; border-bottom: 1px solid rgba(255,255,255,.15); }
#tt table { border-collapse: collapse; width: 100%; }
#tt td { padding: 1px 5px; font-family: monospace; }
#tt td.lbl  { color: #94b8f0; }
#tt td.val  { text-align: right; }
#tt td.nat  { text-align: right; color: #adb5bd; }
#tt th { font-size: 0.65rem; color: #6c757d; padding: 0 5px 2px; text-align: right; font-weight: normal; }

/* SVG chart */
#chart-svg { width: 100%; height: 150px; display: block; }
.ax { font-size: 10px; fill: #6c757d; font-family: monospace; }
</style>
</head>
<body>

<h1>SAE Steering Dashboard</h1>
<p class="meta" id="meta-line"></p>

<div class="controls">
  <label for="m-sel">Method:</label>
  <select id="m-sel"></select>
  <span class="method-desc" id="m-desc"></span>
  <span id="cf-wrap" style="display:none;gap:6px;align-items:center">
    <label for="cf-sel">Color by:</label>
    <select id="cf-sel"></select>
  </span>
</div>

<div class="card">
  <div class="card-title">Tokens</div>
  <div class="legend">
    <div class="legend-item"><div class="swatch" style="background:#e9ecef;border-color:transparent"></div>BOS</div>
    <div class="legend-item"><div class="swatch" style="background:#f0f4ff;border:1.5px solid #93b4ef"></div>Prompt</div>
    <div class="legend-item"><div class="swatch" style="background:#f9f9f9"></div>Inactive</div>
    <div class="legend-item"><div class="swatch" style="background:#22c55e"></div>Active · natural (no intervention)</div>
    <div class="legend-item"><div class="swatch" style="background:#e07b39"></div>Active · intervened</div>
  </div>
  <div id="token-display"></div>
</div>

<div class="card">
  <div class="card-title">Feature activation over sequence</div>
  <svg id="chart-svg" xmlns="http://www.w3.org/2000/svg"></svg>
</div>

<div id="tt"></div>

<script>
const D = __DATA_JSON__;

// ── colour palette for chart lines ─────────────────────────────────────────
const PALETTE = ['#4361ee','#e63946','#2ec4b6','#ff9f1c','#a663cc','#06d6a0'];
function fidColor(idx) { return PALETTE[idx % PALETTE.length]; }

// ── heat scales ────────────────────────────────────────────────────────────
// Orange: active + intervention fired
function heatOrange(v, mx) {
  if (!mx || v <= 0) return '#f9f9f9';
  const t = Math.min(v / mx, 1);
  return `rgb(${Math.round(249+t*(224-249))},${Math.round(249+t*(123-249))},${Math.round(249+t*(57-249))})`;
}
// Green: active + no intervention (natural)
function heatGreen(v, mx) {
  if (!mx || v <= 0) return '#f9f9f9';
  const t = Math.min(v / mx, 1);
  return `rgb(${Math.round(249+t*(34-249))},${Math.round(249+t*(197-249))},${Math.round(249+t*(94-249))})`;
}
function contrastColor(v, mx, isGreen) {
  return (mx && v / mx > 0.55) ? '#fff' : '#1a1a2e';
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── feature max across ALL methods (stable colour scale) ───────────────────
const FMAX = {};
for (const fid of D.feature_ids) {
  let mx = 0;
  for (const m of D.methods)
    for (const e of D.results[m].feature_log) {
      const fd = e[fid];
      if (fd && fd.v > mx) mx = fd.v;
      if (fd && fd.n > mx) mx = fd.n;
    }
  FMAX[fid] = mx;
}

// ── state ──────────────────────────────────────────────────────────────────
let curMethod = D.methods[0];
let colorFid  = D.feature_ids[0];

// ── populate method dropdown ───────────────────────────────────────────────
const mSel = document.getElementById('m-sel');
D.methods.forEach(m => {
  const o = document.createElement('option'); o.value = m; o.textContent = m;
  mSel.appendChild(o);
});
mSel.addEventListener('change', () => { curMethod = mSel.value; refreshDesc(); render(); });

// ── populate colour-by dropdown (only when >1 feature) ────────────────────
if (D.feature_ids.length > 1) {
  const wrap = document.getElementById('cf-wrap');
  wrap.style.display = 'flex';
  const cfSel = document.getElementById('cf-sel');
  D.feature_ids.forEach(fid => {
    const o = document.createElement('option'); o.value = fid;
    o.textContent = `Feature ${fid}`; cfSel.appendChild(o);
  });
  cfSel.addEventListener('change', () => { colorFid = parseInt(cfSel.value); render(); });
}

function refreshDesc() {
  document.getElementById('m-desc').textContent = D.method_descriptions[curMethod] || '';
}
document.getElementById('meta-line').textContent =
  `model: ${D.model_name}  |  hook: ${D.hook_name}  |  features: [${D.feature_ids.join(', ')}]`;
refreshDesc();

// ── tooltip ────────────────────────────────────────────────────────────────
const tt = document.getElementById('tt');
function showTT(e, tokens, log, idx) {
  const isBos    = idx === 0;
  const isPrompt = !isBos && idx < D.n_prompt_tokens;
  const kind     = isBos ? 'BOS' : isPrompt ? 'prompt' : 'generated';

  let rows = `<tr><th></th><th>post-op</th><th>natural</th></tr>`;
  for (const fid of D.feature_ids) {
    const fd = (log[idx] || {})[fid];
    const v  = fd ? fd.v : 0;
    const n  = fd ? fd.n : 0;
    const fired = Math.abs(v - n) > 1e-4;
    const marker = fired ? ' ⚡' : '';
    rows += `<tr>
      <td class="lbl">f${fid}${marker}</td>
      <td class="val">${v.toFixed(4)}</td>
      <td class="nat">${n.toFixed(4)}</td>
    </tr>`;
  }
  tt.innerHTML = `<div class="tt-head">token ${idx} · ${kind}</div><table>${rows}</table>`;
  tt.style.display = 'block';
  moveTT(e);
}
function moveTT(e) {
  tt.style.left = (e.clientX + 14) + 'px';
  tt.style.top  = (e.clientY -  6) + 'px';
}
document.addEventListener('mousemove', e => { if (tt.style.display !== 'none') moveTT(e); });

// ── token colour logic ─────────────────────────────────────────────────────
function tokenStyle(entry, fid, mx) {
  const fd = entry ? entry[fid] : null;
  if (!fd || fd.v <= 0) return { bg: '#f9f9f9', fg: '#1a1a2e' };
  const fired = Math.abs(fd.v - fd.n) > 1e-4;
  if (fired) {
    const bg = heatOrange(fd.v, mx);
    return { bg, fg: contrastColor(fd.v, mx) };
  } else {
    const bg = heatGreen(fd.v, mx);
    return { bg, fg: contrastColor(fd.v, mx) };
  }
}

// ── render tokens ──────────────────────────────────────────────────────────
function render() {
  const { tokens, feature_log } = D.results[curMethod];
  const mx  = FMAX[colorFid] || 1;
  const box = document.getElementById('token-display');
  box.innerHTML = '';

  tokens.forEach((tok, idx) => {
    const isBos  = idx === 0;
    const isPmt  = !isBos && idx < D.n_prompt_tokens;
    const span   = document.createElement('span');
    const display = (tok === '' || tok === ' ') ? (tok === '' ? '·' : tok) : tok;
    span.textContent = display;
    span.className   = 'tok' + (isBos ? ' is-bos' : isPmt ? ' is-prompt' : '');

    if (!isBos) {
      const { bg, fg } = tokenStyle(feature_log[idx], colorFid, mx);
      span.style.background = bg;
      span.style.color      = fg;
    }

    span.addEventListener('mouseenter', e => showTT(e, tokens, feature_log, idx));
    span.addEventListener('mouseleave', () => { tt.style.display = 'none'; });
    box.appendChild(span);
  });

  renderChart();
}

// ── SVG chart ──────────────────────────────────────────────────────────────
function renderChart() {
  const { tokens, feature_log } = D.results[curMethod];
  const n   = tokens.length;
  const svg = document.getElementById('chart-svg');

  const W = 900, H = 150;
  const PL = 48, PR = 12, PT = 10, PB = 28;
  const cW = W - PL - PR, cH = H - PT - PB;

  function xOf(i) { return n > 1 ? PL + (i / (n - 1)) * cW : PL + cW / 2; }
  const yMax = Math.max(1, ...D.feature_ids.map(fid => FMAX[fid] || 0));
  function yOf(v) { return PT + cH - (yMax ? Math.min(v / yMax, 1) : 0) * cH; }

  // One polyline per feature (post-op values)
  const lines = D.feature_ids.map((fid, fi) => {
    const color = fidColor(fi);
    const pts   = feature_log.map((e, i) => {
      const v = (e && e[fid]) ? e[fid].v : 0;
      return `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`;
    }).join(' ');
    return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.8" opacity="0.9"/>`;
  }).join('\n');

  // Natural value dotted polyline for each feature
  const natLines = D.feature_ids.map((fid, fi) => {
    const color = fidColor(fi);
    const pts   = feature_log.map((e, i) => {
      const v = (e && e[fid]) ? e[fid].n : 0;
      return `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`;
    }).join(' ');
    return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1" stroke-dasharray="3,3" opacity="0.45"/>`;
  }).join('\n');

  // Prompt / generated separator
  const sx  = xOf(D.n_prompt_tokens - 1);
  const sep = `<line x1="${sx.toFixed(1)}" y1="${PT}" x2="${sx.toFixed(1)}" y2="${PT+cH}"
    stroke="#adb5bd" stroke-width="1" stroke-dasharray="4,3"/>`;

  // Axes
  const xaxis = `<line x1="${PL}" y1="${PT+cH}" x2="${W-PR}" y2="${PT+cH}" stroke="#dee2e6" stroke-width="1"/>`;
  const yaxis = `<line x1="${PL}" y1="${PT}"    x2="${PL}"   y2="${PT+cH}" stroke="#dee2e6" stroke-width="1"/>`;

  // Y ticks
  const yticks = [0, 0.5, 1].map(t => {
    const v = t * yMax, y = yOf(v);
    return `<line x1="${PL-3}" y1="${y.toFixed(1)}" x2="${PL}" y2="${y.toFixed(1)}" stroke="#adb5bd" stroke-width="1"/>
    <text x="${(PL-5).toFixed(1)}" y="${(y+3.5).toFixed(1)}" class="ax" text-anchor="end">${v.toFixed(1)}</text>`;
  }).join('');

  // Region labels
  const lblPrompt = `<text x="${(PL+4).toFixed(1)}" y="${(PT+cH-4).toFixed(1)}" class="ax" fill="#93b4ef">← prompt</text>`;
  const lblGen    = `<text x="${(sx+5).toFixed(1)}"  y="${(PT+cH-4).toFixed(1)}" class="ax">generated →</text>`;

  // Legend (solid = post-op, dashed = natural)
  const legend = D.feature_ids.map((fid, fi) => {
    const x = PL + 10 + fi * 150, y = H - 6;
    return `<line x1="${x}" y1="${y}" x2="${x+14}" y2="${y}" stroke="${fidColor(fi)}" stroke-width="2"/>
    <text x="${x+18}" y="${y+3.5}" class="ax">f${fid} post-op</text>
    <line x1="${x+80}" y1="${y}" x2="${x+94}" y2="${y}" stroke="${fidColor(fi)}" stroke-width="1" stroke-dasharray="3,3" opacity="0.55"/>
    <text x="${x+98}" y="${y+3.5}" class="ax" opacity="0.55">natural</text>`;
  }).join('\n');

  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.innerHTML = [xaxis, yaxis, yticks, sep, natLines, lines, legend, lblPrompt, lblGen].join('\n');
}

// ── init ──────────────────────────────────────────────────────────────────
render();
</script>
</body>
</html>"""

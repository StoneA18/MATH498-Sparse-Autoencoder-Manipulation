"""
feature_steering.py — Core module for LLM feature analysis and steering.

Inspired by Anthropic's "Golden Gate Claude" experiment, this module lets you:
  1. Find which internal "features" a model activates for a given text.
  2. Clamp specific features to fixed values during generation.
  3. Observe how those clamps change what the model says.

Background — what is a feature?
  Transformer models store information in a "residual stream": a vector that flows
  through every layer, accumulating updates from attention and MLP blocks. This
  vector lives in a high-dimensional space, and many distinct concepts are
  superimposed on top of each other ("superposition").

  Sparse Autoencoders (SAEs) are a tool from mechanistic interpretability research
  that decompose these dense vectors into a much larger set of sparse, roughly
  interpretable "features". Each feature is a direction in activation space that
  (ideally) corresponds to one concept the model can represent — e.g. "this token
  is part of a city name" or "the text is about biology".

  Once we have those features, we can *clamp* them: overwrite a feature's value
  mid-forward-pass, forcing the model to behave as if that concept is always present.
  That's how Golden Gate Claude worked — Anthropic clamped a "Golden Gate Bridge"
  feature so the model kept relating everything back to that landmark.

Supported models (see MODELS dict below):
  "gpt2-small"   GPT-2 Small (124M, raw LM) — fast, good for experimentation
  "gemma-2b-it"  Gemma 2B Instruction-Tuned (2B) — can actually answer questions

Feature lookup:
  Human-written feature interpretations live at Neuronpedia:
  https://neuronpedia.org/<neuronpedia_id>/<feature_id>
  (each ModelConfig below has the right neuronpedia_id for its model+SAE)
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from transformer_lens import HookedTransformer
from sae_lens import SAE


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Everything needed to load a model and optionally a paired SAE.

    Fields:
        display_name        Human-readable name shown in the REPL.
        tl_name             TransformerLens model identifier.
        tl_kwargs           Extra kwargs forwarded to HookedTransformer.from_pretrained().
        is_chat             True for instruction-tuned models that expect a chat format.
        chat_template       Python format string with one {prompt} placeholder.
        default_max_tokens  Default token budget for generate/compare on this model.

    SAE fields (all optional — omit for generation-only models):
        sae_release      sae_lens release name.
        hook_point       Residual-stream hook the SAE was trained on.
        neuronpedia_id   Used to build Neuronpedia URLs: .org/<id>/<feature_id>
    """
    display_name:       str
    tl_name:            str
    tl_kwargs:          Dict            = field(default_factory=dict)
    is_chat:            bool            = False
    chat_template:      Optional[str]   = None
    default_max_tokens: int             = 150
    sae_release:        Optional[str]   = None   # sae_lens release name
    sae_id:             Optional[str]   = None   # SAE identifier within the release
    # hook_point is the TransformerLens hook name the SAE attaches to.
    # If left None, it is read from cfg['hook_name'] after the SAE loads
    # (gemma-scope SAEs self-report their hook point in their config file).
    hook_point:         Optional[str]   = None
    neuronpedia_id:     Optional[str]   = None   # base URL segment for feature lookup

    @property
    def has_sae(self) -> bool:
        return self.sae_release is not None


MODELS: Dict[str, ModelConfig] = {
    # ── GPT-2 Small ──────────────────────────────────────────────────────────
    # Tiny raw language model (not instruction-tuned). Useful for understanding
    # the steering mechanics at low cost, but cannot reliably answer questions.
    # ── GPT-2 Small (124M) ───────────────────────────────────────────────────
    # Tiny raw LM. Fast and good for understanding the steering mechanics,
    # but cannot reliably answer factual questions.
    "gpt2-small": ModelConfig(
        display_name   = "GPT-2 Small (124M, raw LM)",
        tl_name        = "gpt2",
        tl_kwargs      = dict(
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            refactor_factored_attn_matrices=True,
        ),
        sae_release    = "gpt2-small-res-jb",
        sae_id         = "blocks.8.hook_resid_pre",   # jb format: sae_id == hook_point
        hook_point     = "blocks.8.hook_resid_pre",
        neuronpedia_id = "gpt2-small/8-res-jb",
    ),

    # ── Gemma 3 270M Instruction-Tuned (270M) ────────────────────────────────
    # Much smaller than Gemma 2B but still instruction-tuned, so it can answer
    # questions and follow directions. ~500 MB download. SAEs from GemmaScope.
    # Neuronpedia: https://neuronpedia.org/gemma-3-270m-it/5-gemmascope-2-res-16k
    "gemma-3-270m-it": ModelConfig(
        display_name        = "Gemma 3 270M Instruction-Tuned (270M, chat model)",
        tl_name             = "google/gemma-3-270m-it",
        tl_kwargs           = dict(dtype=torch.bfloat16),
        is_chat             = True,
        chat_template       = (
            "<start_of_turn>user\n"
            "{prompt}<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
        default_max_tokens  = 25,
        sae_release         = "gemma-scope-2-270m-it-res",
        # GemmaScope SAE IDs use layer/width/sparsity naming.
        # layer_5 = layer 5, residual-post — matches TL hook name below.
        sae_id              = "layer_5_width_16k_l0_medium",
        hook_point          = "blocks.5.hook_resid_post",
        neuronpedia_id      = "gemma-3-270m-it/5-gemmascope-2-res-16k",
    ),

    # ── Gemma 2B Instruction-Tuned (2B) ──────────────────────────────────────
    # Full-size chat model. Good answers but slow on CPU (~1-3 tok/s).
    # ~5 GB download. SAE by Joseph Bloom, 16,384 features on layer 12.
    "gemma-2b-it": ModelConfig(
        display_name        = "Gemma 2B Instruction-Tuned (2B, chat model)",
        tl_name             = "gemma-2b-it",
        # bfloat16: same memory as float16 (~4 GB) but natively supported on
        # modern CPUs; float16 falls back to float32 ops internally on CPU.
        tl_kwargs           = dict(dtype=torch.bfloat16),
        is_chat             = True,
        chat_template       = (
            "<start_of_turn>user\n"
            "{prompt}<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
        default_max_tokens  = 50,
        sae_release         = "gemma-2b-it-res-jb",
        sae_id              = "blocks.12.hook_resid_post",
        hook_point          = "blocks.12.hook_resid_post",
        neuronpedia_id      = "gemma-2b/12-res-jb",
    ),
}

DEFAULT_MODEL = "gpt2-small"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model_and_sae(
    config_name: str = DEFAULT_MODEL,
    device: str = "cpu",
) -> Tuple[HookedTransformer, SAE, ModelConfig]:
    """
    Load a model+SAE pair by config name.

    Args:
        config_name  Key from the MODELS dict (e.g. "gpt2-small", "gemma-2b-it").
        device       Torch device string ("cpu" or "cuda").

    Returns:
        model    — HookedTransformer ready for analysis and generation
        sae      — Sparse Autoencoder trained on model's residual stream
        config   — The ModelConfig used (callers need hook_point etc. from it)
    """
    if config_name not in MODELS:
        known = ", ".join(f'"{k}"' for k in MODELS)
        raise ValueError(f"Unknown model {config_name!r}. Known: {known}")

    config = MODELS[config_name]

    print(f"Loading {config.display_name}...")
    model = HookedTransformer.from_pretrained(config.tl_name, **config.tl_kwargs)
    model = model.to(device)
    model.eval()

    print(f"Loading SAE ({config.sae_id})...")
    # sae_lens >= 6.x: from_pretrained_with_cfg_and_sparsity returns (sae, cfg_dict, sparsity).
    sae, cfg, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=config.sae_release,
        sae_id=config.sae_id,
        device=device,
    )
    sae.eval()

    # GemmaScope SAEs self-report their TransformerLens hook name in the cfg.
    # For jb-format SAEs the hook_point is set explicitly in ModelConfig.
    hook_point = config.hook_point or cfg.get("hook_name")
    if hook_point is None:
        raise RuntimeError(
            f"Could not determine hook_point for {config_name!r}. "
            "Set hook_point in ModelConfig or ensure the SAE cfg contains 'hook_name'."
        )

    # Return a copy of the config with hook_point filled in (important when it
    # was None and we just resolved it from the SAE cfg).
    import dataclasses
    resolved_config = dataclasses.replace(config, hook_point=hook_point)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model:  {config.display_name} — {n_params:,} parameters")
    print(f"  SAE:    {cfg['d_in']} residual dims -> {cfg['d_sae']} features")
    print(f"  Hook:   {hook_point}")
    print()
    return model, sae, resolved_config


def format_prompt(text: str, config: ModelConfig) -> str:
    """
    Wrap `text` in the model's chat format if it is an instruction-tuned model.

    For raw LMs (is_chat=False) the text is returned unchanged.
    For chat models the text is inserted into the chat_template so the model
    knows to treat it as a user message and generate an assistant reply.
    """
    if config.is_chat and config.chat_template:
        return config.chat_template.format(prompt=text)
    return text


# ---------------------------------------------------------------------------
# Feature Analysis
# ---------------------------------------------------------------------------

def get_feature_activations(
    text: str,
    model: HookedTransformer,
    sae: SAE,
    hook_point: str,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Run text through the model and return SAE feature activations.

    For each token position, the SAE encoder takes the 768-dim residual stream
    vector and outputs a 24,576-dim sparse vector. Most values are 0; the nonzero
    ones are the "features" that fired at that position.

    Returns:
        Tensor of shape [seq_len, d_sae]
        result[i, j] = activation of feature j at token position i
    """
    tokens = model.to_tokens(text)  # [1, seq_len]

    # Run the model and grab only the residual stream at our hook point.
    # `names_filter` avoids caching every intermediate activation (saves memory).
    _, cache = model.run_with_cache(tokens, names_filter=hook_point)
    resid = cache[hook_point].to(device)  # [1, seq_len, d_model]

    with torch.no_grad():
        feature_acts = sae.encode(resid)  # [1, seq_len, d_sae]

    return feature_acts[0]  # drop batch dim → [seq_len, d_sae]


def top_features_for_text(
    text: str,
    model: HookedTransformer,
    sae: SAE,
    hook_point: str,
    k: int = 10,
    device: str = "cpu",
    aggregate: str = "max",
) -> List[Tuple[int, float]]:
    """
    Find the k features most strongly activated by the given text.

    Args:
        text:       Input text to analyze.
        k:          How many top features to return.
        aggregate:  How to combine activations across token positions:
                    "max"  — peak activation across tokens (catches concept triggers)
                    "mean" — average activation (captures sustained themes)

    Returns:
        List of (feature_index, activation_value) sorted from highest to lowest.
    """
    feature_acts = get_feature_activations(text, model, sae, hook_point, device)
    # feature_acts: [seq_len, d_sae]

    if aggregate == "max":
        scores, _ = feature_acts.max(dim=0)   # [d_sae] — peak per feature
    else:
        scores = feature_acts.mean(dim=0)     # [d_sae] — mean per feature

    top_vals, top_idx = scores.topk(k)
    return [(int(i), float(v)) for i, v in zip(top_idx, top_vals)]


def token_feature_activations(
    text: str,
    feature_id: int,
    model: HookedTransformer,
    sae: SAE,
    hook_point: str,
    device: str = "cpu",
) -> List[Tuple[str, float]]:
    """
    Show how strongly one specific feature activates at each token in text.

    Useful for understanding what a feature responds to: does it fire on a specific
    word, on a syntactic position, on a semantic category? Running the same feature
    across many examples builds intuition about what concept it tracks.

    Returns:
        List of (token_string, activation_value) in order of token position.
    """
    token_strs = model.to_str_tokens(text)                    # e.g. ["The", " cat", " sat"]
    feature_acts = get_feature_activations(text, model, sae, hook_point, device)
    # feature_acts: [seq_len, d_sae]

    feature_col = feature_acts[:, feature_id]                 # [seq_len] — just this feature
    return [(tok, float(val)) for tok, val in zip(token_strs, feature_col)]


# ---------------------------------------------------------------------------
# HTML Exploration Report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Feature Exploration</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 16px 20px;
         background: #f5f6f8; color: #1a1a2e; }
  h1 { margin: 0 0 4px; font-size: 1.15rem; font-weight: 600; }
  .meta { color: #6c757d; font-size: 0.82rem; margin-bottom: 14px; font-family: monospace; }

  /* ---- search bar ---- */
  .search-bar { display: flex; gap: 8px; align-items: center; margin-bottom: 14px;
                flex-wrap: wrap; }
  .search-bar label { font-size: 0.88rem; font-weight: 500; }
  .search-bar input { padding: 5px 9px; border: 1px solid #ced4da; border-radius: 4px;
                      font-size: 0.88rem; width: 140px; }
  .search-bar input:focus { outline: 2px solid #4361ee; border-color: transparent; }
  .search-bar button { padding: 5px 13px; background: #4361ee; color: #fff;
                       border: none; border-radius: 4px; cursor: pointer; font-size: 0.88rem; }
  .search-bar button:hover { background: #3451d1; }
  .err { color: #d62839; font-size: 0.82rem; }

  /* ---- detail panel ---- */
  #detail { display: none; background: #fff; border: 1px solid #dee2e6; border-radius: 6px;
            padding: 14px 16px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
  #detail h2 { margin: 0 0 10px; font-size: 0.95rem; }
  #detail .close-btn { float: right; background: none; border: none; cursor: pointer;
                       font-size: 1rem; color: #6c757d; line-height: 1; }
  #detail .close-btn:hover { color: #1a1a2e; }
  #dtable { border-collapse: collapse; width: 100%; font-size: 0.82rem; }
  #dtable th { background: #f1f3f5; padding: 5px 10px; text-align: left;
               border-bottom: 2px solid #dee2e6; font-weight: 600; }
  #dtable td { padding: 3px 10px; border-bottom: 1px solid #f1f3f5; font-family: monospace; }
  #dtable tr.zero td { color: #adb5bd; }
  .bar-wrap { width: 140px; }
  .bar { display: inline-block; height: 11px; background: #4361ee; border-radius: 2px;
         vertical-align: middle; min-width: 0; }

  /* ---- main token table ---- */
  #main-wrap { overflow-x: auto; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
  #main { border-collapse: collapse; white-space: nowrap; font-size: 0.8rem;
          background: #fff; width: 100%; }
  #main thead th { background: #1a1a2e; color: #e9ecef; padding: 6px 10px; text-align: left;
                   position: sticky; top: 0; z-index: 2; }
  #main thead th.pos-h { width: 40px; }
  #main thead th.tok-h { min-width: 80px; }
  #main td { border-bottom: 1px solid #e9ecef; padding: 4px 6px; vertical-align: top; }
  #main td.pos { color: #adb5bd; font-size: 0.75rem; text-align: right; width: 36px;
                 background: #f8f9fa; position: sticky; left: 0; z-index: 1; }
  #main td.tok { font-family: monospace; font-weight: 600; background: #f8f9fa;
                 position: sticky; left: 36px; z-index: 1; padding: 4px 10px; min-width: 70px; }
  .fc { display: inline-block; padding: 2px 7px; border-radius: 3px; cursor: pointer;
        font-family: monospace; line-height: 1.5; transition: filter .1s; }
  .fc:hover { filter: brightness(.88); }
  .fc.active { outline: 2px solid #1a1a2e; outline-offset: 1px; }
  .fc small { opacity: .75; font-size: .9em; }
</style>
</head>
<body>
<h1>Feature Exploration</h1>
<p class="meta" id="meta"></p>

<div class="search-bar">
  <label for="fsearch">Show feature:</label>
  <input id="fsearch" type="number" min="0" placeholder="feature ID" />
  <button onclick="doSearch()">Show</button>
  <span class="err" id="serr"></span>
</div>

<div id="detail">
  <button class="close-btn" onclick="closeDetail()" title="Close">&#x2715;</button>
  <h2>Feature <span id="did"></span> &mdash; activation per token</h2>
  <table id="dtable">
    <thead><tr><th>#</th><th>Token</th><th>Activation</th><th class="bar-wrap">Magnitude</th></tr></thead>
    <tbody id="dbody"></tbody>
  </table>
</div>

<div id="main-wrap">
  <table id="main">
    <thead id="mhead"></thead>
    <tbody id="mbody"></tbody>
  </table>
</div>

<script>
const D = __DATA_JSON__;

/* pre-compute per-feature max for colour scaling */
const fMax = {};
for (const [k, vs] of Object.entries(D.featureActivations))
  fMax[k] = Math.max(0, ...vs);

/* white -> indigo heat colour */
function heatRgb(val, maxVal) {
  if (!maxVal) return '#f8f9fa';
  const t = Math.min(val / maxVal, 1);
  const r = Math.round(255 + t * (67  - 255));
  const g = Math.round(255 + t * (97  - 255));
  const b = Math.round(255 + t * (238 - 255));
  return `rgb(${r},${g},${b})`;
}

function textColor(val, maxVal) {
  return (maxVal && val / maxVal > 0.55) ? '#fff' : '#1a1a2e';
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ---- build header ---- */
let hh = '<tr><th class="pos-h">#</th><th class="tok-h">Token</th>';
for (let r = 1; r <= D.topN; r++) hh += `<th>Rank&nbsp;${r}</th>`;
hh += '</tr>';
document.getElementById('mhead').innerHTML = hh;

/* ---- build body ---- */
let bh = '';
for (let i = 0; i < D.tokens.length; i++) {
  bh += `<tr><td class="pos">${i}</td><td class="tok">${escHtml(D.tokens[i])}</td>`;
  for (let r = 0; r < D.topN; r++) {
    const [fid, val] = D.topFeatures[i][r] || [0, 0];
    const mx = fMax[String(fid)] || 1;
    const bg = heatRgb(val, mx);
    const fg = textColor(val, mx);
    bh += `<td><span class="fc" style="background:${bg};color:${fg}"
      data-fid="${fid}" onclick="showFeature(${fid})"
      title="feature ${fid} = ${val.toFixed(4)}">${fid} <small>(${val.toFixed(2)})</small></span></td>`;
  }
  bh += '</tr>';
}
document.getElementById('mbody').innerHTML = bh;

/* ---- meta line ---- */
document.getElementById('meta').textContent =
  `model: ${D.modelName}  |  hook: ${D.hookPoint}  |  ` +
  `${D.tokens.length} tokens  |  ${Object.keys(D.featureActivations).length} unique features in top ${D.topN}`;

/* ---- detail panel ---- */
let activeId = null;
function showFeature(fid) {
  const key = String(fid);
  if (!D.featureActivations[key]) {
    document.getElementById('serr').textContent =
      `Feature ${fid} never reached top ${D.topN} — not included in this report.`;
    return;
  }
  document.getElementById('serr').textContent = '';
  activeId = fid;

  document.querySelectorAll('.fc').forEach(el =>
    el.classList.toggle('active', Number(el.dataset.fid) === fid));

  document.getElementById('did').textContent = fid;
  const vals = D.featureActivations[key];
  const mx = fMax[key] || 1;
  const BAR = 130;
  let rows = '';
  for (let i = 0; i < D.tokens.length; i++) {
    const v = vals[i];
    const w = Math.round((v / mx) * BAR);
    rows += `<tr class="${v === 0 ? 'zero' : ''}">
      <td>${i}</td><td>${escHtml(D.tokens[i])}</td>
      <td>${v.toFixed(4)}</td>
      <td class="bar-wrap"><span class="bar" style="width:${w}px"></span></td>
    </tr>`;
  }
  document.getElementById('dbody').innerHTML = rows;
  const panel = document.getElementById('detail');
  panel.style.display = 'block';
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeDetail() {
  document.getElementById('detail').style.display = 'none';
  document.querySelectorAll('.fc.active').forEach(el => el.classList.remove('active'));
  activeId = null;
}

function doSearch() {
  const v = parseInt(document.getElementById('fsearch').value, 10);
  if (isNaN(v)) { document.getElementById('serr').textContent = 'Enter an integer feature ID.'; return; }
  showFeature(v);
}

document.getElementById('fsearch').addEventListener('keydown', e => {
  if (e.key === 'Enter') doSearch();
});
</script>
</body>
</html>"""


def generate_exploration_html(
    text: str,
    model: HookedTransformer,
    sae: SAE,
    hook_point: str,
    top_n: int = 10,
    device: str = "cpu",
    model_name: str = "",
) -> str:
    """
    Run text through the model+SAE and return a self-contained HTML report.

    The report shows a token-by-token table of the top-N activated features,
    colour-coded by activation strength.  Clicking any feature cell opens a
    detail panel with that feature's activation at every token position.
    Only features that appear in the top-N for at least one token are included,
    keeping the file size manageable.

    Args:
        text:        Input text to analyse.
        model:       Loaded HookedTransformer.
        sae:         SAE to use for feature decomposition.
        hook_point:  TransformerLens hook name the SAE was trained on.
        top_n:       Number of top features to show per token.  Default: 10.
        device:      Torch device string.
        model_name:  Display name shown in the report header.

    Returns:
        A self-contained HTML string ready to write to a .html file.
    """
    import json

    feature_acts = get_feature_activations(text, model, sae, hook_point, device)
    # feature_acts: [seq_len, d_sae]

    token_strs = model.to_str_tokens(text)
    seq_len, d_sae = feature_acts.shape
    actual_top_n = min(top_n, d_sae)

    top_vals, top_idx = feature_acts.topk(actual_top_n, dim=1)
    # top_vals, top_idx: [seq_len, actual_top_n]

    # Collect all feature IDs that appear in any token's top-N
    featured_ids: set = set()
    for i in range(seq_len):
        for fid in top_idx[i].tolist():
            featured_ids.add(int(fid))

    # Per-token top-N list  [[fid, val], ...]
    top_features = [
        [[int(top_idx[i, r]), float(top_vals[i, r])] for r in range(actual_top_n)]
        for i in range(seq_len)
    ]

    # Full activation vector for every featured feature
    feature_activations = {
        str(fid): [float(v) for v in feature_acts[:, fid].tolist()]
        for fid in sorted(featured_ids)
    }

    data = {
        "tokens":             [str(t) for t in token_strs],
        "topFeatures":        top_features,
        "featureActivations": feature_activations,
        "topN":               actual_top_n,
        "modelName":          model_name,
        "hookPoint":          hook_point,
    }

    # Embed data; escape </script> to prevent early tag close
    data_json = json.dumps(data, separators=(",", ":")).replace("</script>", r"<\/script>")
    return _HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


# ---------------------------------------------------------------------------
# Feature Steering
# ---------------------------------------------------------------------------

class FeatureSteerer:
    """
    Steers GPT-2's text generation by clamping specific SAE features.

    How clamping works (step by step):
      1. We register a hook on the residual stream at layer 8.
      2. On every forward pass, the hook intercepts the residual stream.
      3. It encodes that stream through the SAE to get sparse feature activations.
      4. It overwrites the desired feature slots with the clamped values.
      5. It decodes back to residual stream space (preserving the SAE error term).
      6. The model continues its forward pass with the modified stream.

    The "SAE error term" (step 5) is the part of the residual stream that the SAE
    can't reconstruct — keeping it ensures we make minimal changes and don't throw
    away information the SAE didn't learn to represent.

    Typical activation magnitudes: features that are "off" = 0.0, active features
    typically range from ~1 to ~30. Setting a clamp value of 20–40 is a strong
    activation; very high values (>100) can make outputs incoherent.

    Example:
        steerer = FeatureSteerer(model, sae, hook_point)
        steerer.clamp(feature_id=683, value=25.0)
        print(steerer.generate("Tell me about yourself"))
        steerer.unclamp_all()
    """

    def __init__(
        self,
        model: HookedTransformer,
        sae: SAE,
        hook_point: str,
        device: str = "cpu",
    ):
        self.model = model
        self.sae = sae
        self.hook_point = hook_point
        self.device = device
        # Maps feature_id → operation tuple.  Empty dict = no steering.
        # ('clamp', value)               — set to value
        # ('add',   value)               — add value
        # ('scale', factor)              — multiply by factor
        # ('cond_dist', prob, dist, ...) — per-token stochastic set (see cond_dist())
        self.clamped_features: Dict[int, tuple] = {}

    # ------------------------------------------------------------------
    # Managing clamps
    # ------------------------------------------------------------------

    def clamp(self, feature_id: int, value: float = 20.0):
        """Force feature_id to always equal value during generation."""
        self.clamped_features[feature_id] = ('clamp', value)

    def add(self, feature_id: int, value: float = 10.0):
        """Add value to the activation of feature_id during generation."""
        self.clamped_features[feature_id] = ('add', value)

    def scale(self, feature_id: int, factor: float = 2.0):
        """Multiply the activation of feature_id by factor during generation."""
        self.clamped_features[feature_id] = ('scale', factor)

    def cond_dist(
        self,
        feature_id: int,
        prob: float,
        dist: str,
        *dist_params: float,
    ):
        """
        At each token position, independently set feature_id to a sampled value
        with probability `prob`; otherwise leave it untouched.

        Args:
            feature_id:  Feature to steer.
            prob:        Probability (0–1) of applying the intervention per token.
            dist:        Distribution to sample from.  Supported values:
                           "normal"  — dist_params: mean, std
                           "uniform" — dist_params: low, high
            *dist_params: Parameters for the chosen distribution (see above).

        Example:
            steerer.cond_dist(18493, 0.5, "normal", 40, 10)
            # Each token: 50 % chance feature 18493 is set to Normal(40, 10)
        """
        supported = ("normal", "uniform")
        if dist not in supported:
            raise ValueError(f"Unknown dist {dist!r}. Supported: {supported}")
        self.clamped_features[feature_id] = ('cond_dist', prob, dist, *dist_params)

    def unclamp(self, feature_id: int):
        """Remove the clamp on a single feature (returns to normal)."""
        self.clamped_features.pop(feature_id, None)

    def unclamp_all(self):
        """Remove all feature clamps — model returns to normal behavior."""
        self.clamped_features.clear()

    def list_clamps(self) -> Dict[int, Tuple[str, float]]:
        """Return the current clamp settings."""
        return dict(self.clamped_features)

    # ------------------------------------------------------------------
    # Hook construction
    # ------------------------------------------------------------------

    def _build_hook(self):
        """
        Return a hook function that intercepts and modifies the residual stream.

        We snapshot the clamp dict at build time so the hook has a stable
        reference for the duration of the generation call.
        """
        clamped = dict(self.clamped_features)   # snapshot
        sae = self.sae

        def hook_fn(resid, hook):
            # resid: [batch, seq_len, d_model] — the residual stream tensor
            with torch.no_grad():
                # Step 1: Encode into sparse feature space.
                feature_acts = sae.encode(resid)          # [batch, seq_len, d_sae]

                # Step 2: Compute what the SAE can reconstruct normally.
                original_recon = sae.decode(feature_acts) # [batch, seq_len, d_model]

                # Step 3: Preserve the error — what the SAE can't explain.
                # This stays unchanged so our intervention is targeted.
                sae_error = resid - original_recon        # [batch, seq_len, d_model]

                # Step 4: Apply the manipulations to the clamped features at every token position.
                for feat_id, entry in clamped.items():
                    op = entry[0]
                    if op == 'clamp':
                        feature_acts[:, :, feat_id] = entry[1]
                    elif op == 'add':
                        feature_acts[:, :, feat_id] += entry[1]
                    elif op == 'scale':
                        feature_acts[:, :, feat_id] *= entry[1]
                    elif op == 'cond_dist':
                        _, prob, dist, *params = entry
                        batch, seq_len = feature_acts.shape[:2]
                        # Bernoulli mask: True where we apply the intervention.
                        mask = torch.bernoulli(
                            torch.full((batch, seq_len), prob, device=feature_acts.device)
                        ).bool()
                        if dist == 'normal':
                            mean, std = params
                            samples = torch.normal(
                                mean=float(mean),
                                std=float(std),
                                size=(batch, seq_len),
                                device=feature_acts.device,
                            )
                        else:  # 'uniform'
                            low, high = params
                            samples = torch.empty(
                                batch, seq_len, device=feature_acts.device
                            ).uniform_(float(low), float(high))
                        feature_acts[:, :, feat_id] = torch.where(
                            mask, samples, feature_acts[:, :, feat_id]
                        )

                # Step 5: Decode the modified features and add back the error.
                new_resid = sae.decode(feature_acts) + sae_error

            return new_resid

        return hook_fn

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 1.0,
        top_p: float = 0.9,
        verbose: bool = True,
    ) -> str:
        """
        Generate text from prompt with all current feature clamps active.

        The steering hook is registered before generation and removed afterward
        (even if generation raises an exception), so the model is always left in
        a clean state.

        TransformerLens's .generate() uses a KV cache internally, which means
        only the newest token's activations pass through the model at each step.
        Our hook fires on each of those new-token passes, so the clamp is applied
        at every generated token — exactly what we want.

        Args:
            prompt:         Text to continue from.
            max_new_tokens: Maximum number of tokens to generate.
            temperature:    Sampling temperature. 1.0 = default; lower = more focused.
            top_p:          Nucleus sampling cutoff. 0.9 = sample from top 90% mass.
            verbose:        Print a note when generating without any clamps.

        Returns:
            The generated continuation (not including the original prompt).
        """
        if verbose and not self.clamped_features:
            print("[note] No features clamped — generating with unmodified model.")

        tokens = self.model.to_tokens(prompt)  # [1, seq_len]

        # For chat models (Gemma IT etc.) pass <end_of_turn> as eos_token_id so
        # generation stops immediately instead of spamming end-of-turn tokens.
        generate_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=False,
        )
        eot_id = self.model.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if eot_id is not None and eot_id != self.model.tokenizer.unk_token_id:
            generate_kwargs["eos_token_id"] = eot_id

        # Register our hook persistently so it fires during every internal
        # forward pass that .generate() makes (one per new token, with KV cache).
        try:
            self.model.add_hook(self.hook_point, self._build_hook())
            with torch.no_grad():
                output = self.model.generate(tokens, **generate_kwargs)
        finally:
            # Always clean up hooks, even on error.
            self.model.reset_hooks()

        # Decode just the newly generated tokens (not the prompt).
        prompt_len = tokens.shape[1]
        new_tokens = output[0, prompt_len:]
        result = self.model.to_string(new_tokens)

        # Chat models (e.g. Gemma IT) emit <end_of_turn> when the assistant turn
        # ends, then repeat it to fill the remaining token budget. Truncate there.
        if "<end_of_turn>" in result:
            result = result.split("<end_of_turn>")[0]

        return result.strip()


# ---------------------------------------------------------------------------
# Custom SAE Training
# ---------------------------------------------------------------------------

# Root directory for all locally-trained SAEs.
# Each SAE is stored as  SAES_ROOT/<name>/cfg.json  +  sae_weights.safetensors
SAES_ROOT = "./custom_saes"

# Hook type → TransformerLens hook name fragment
_HOOK_TYPE_MAP = {
    "resid_post": "hook_resid_post",
    "resid_pre":  "hook_resid_pre",
    "mlp_out":    "hook_mlp_out",
    "attn_out":   "hook_attn_out",
}


def load_sae_from_name(name: str, device: str = "cpu") -> Tuple[SAE, str]:
    """
    Load a locally-trained SAE by name.

    Looks for   SAES_ROOT/<name>/cfg.json  (written by train_custom_sae).

    Args:
        name:    The short name given when the SAE was trained (e.g. "sae1").
        device:  Torch device string.

    Returns:
        (sae, hook_point) — the loaded SAE and the hook name it was trained on.

    Raises:
        FileNotFoundError: if no SAE with that name exists.
        RuntimeError:      if the SAE config does not contain a hook_name.
    """
    import json
    import os
    path = os.path.join(SAES_ROOT, name)
    cfg_file = os.path.join(path, "cfg.json")
    if not os.path.isfile(cfg_file):
        known = [
            d for d in os.listdir(SAES_ROOT)
            if os.path.isfile(os.path.join(SAES_ROOT, d, "cfg.json"))
        ] if os.path.isdir(SAES_ROOT) else []
        hint = f"  Known SAEs: {', '.join(known)}" if known else "  (no SAEs trained yet)"
        raise FileNotFoundError(
            f"No SAE named {name!r} found in {SAES_ROOT!r}.\n{hint}"
        )

    with open(cfg_file) as f:
        cfg_dict = json.load(f)

    # hook_name is stored under metadata in SAELens-trained SAEs.
    metadata   = cfg_dict.get("metadata", {})
    hook_point = metadata.get("hook_name") or cfg_dict.get("hook_name")
    if not hook_point:
        raise RuntimeError(
            f"SAE {name!r} does not report a hook_name in its config."
        )

    sae = SAE.load_from_disk(path, device=device)
    sae.eval()

    d_in  = cfg_dict.get("d_in",  "?")
    d_sae = cfg_dict.get("d_sae", "?")
    print(f"  SAE:  {d_in} -> {d_sae} features")
    print(f"  Hook: {hook_point}")
    return sae, hook_point


def train_custom_sae(
    model: HookedTransformer,
    config_name: str,
    layer: int,
    name: str = "sae",
    hook_type: str = "resid_post",
    expansion_factor: int = 16,
    d_sae: Optional[int] = None,
    activation_fn: str = "topk",
    k: int = 50,
    l1_coefficient: float = 5e-5,
    lr: float = 2e-4,
    training_tokens: int = 500_000,
    train_batch_size_tokens: int = 4096,
    context_size: int = 128,
    dataset_path: str = "NeelNanda/pile-10k",
    device: str = "cpu",
    log_to_wandb: bool = False,
) -> SAE:
    """
    Train a custom Sparse Autoencoder on an already-loaded HookedTransformer.

    The finished SAE is saved to  SAES_ROOT/<name>/  (cfg.json +
    sae_weights.safetensors) and can be reloaded with load_sae_from_name(name).

    Args:
        model:                  A loaded HookedTransformer.  Its d_model sets d_in;
                                the runner reuses these weights via override_model.
        config_name:            Key from the MODELS dict (used for the TL model name).
                                If not in MODELS, treated as a raw TL model name.
        layer:                  Transformer layer to attach the SAE to.
        name:                   Short identifier for this SAE.  Saved to
                                SAES_ROOT/<name>/.  Default: "sae".
        hook_type:              Activation to train on:
                                  "resid_post"  residual stream after layer (default)
                                  "resid_pre"   residual stream before layer
                                  "mlp_out"     MLP block output
                                  "attn_out"    attention block output
        expansion_factor:       d_sae = d_in × expansion_factor (ignored if d_sae set).
        d_sae:                  Dictionary size (features).  Overrides expansion_factor.
        activation_fn:          "topk" (recommended) or "relu".
        k:                      Active features per token for TopK.  Default: 50.
        l1_coefficient:         Sparsity penalty for ReLU.  Default: 5e-5.
        lr:                     AdamW learning rate.  Default: 2e-4.
        training_tokens:        Total tokens to train on.  Default: 500k.
        train_batch_size_tokens: Tokens per gradient step.  Default: 4096.
        context_size:           Sequence length for activation harvesting.  Default: 128.
        dataset_path:           HuggingFace dataset.  Default: "NeelNanda/pile-10k".
        device:                 Torch device string.  Default: "cpu".
        log_to_wandb:           Log to Weights & Biases.  Default: False.

    Returns:
        The trained SAE, already saved to SAES_ROOT/<name>/.

    Raises:
        ImportError:  if sae_lens training utilities are not installed.
        ValueError:   if hook_type or activation_fn is not recognised.
    """
    import glob as _glob
    import math
    import os
    import shutil

    try:
        from sae_lens import (
            LanguageModelSAERunnerConfig,
            LanguageModelSAETrainingRunner,
            LoggingConfig,
            StandardTrainingSAEConfig,
            TopKTrainingSAEConfig,
        )
    except ImportError as exc:
        raise ImportError(
            "SAELens training requires the full sae_lens package. "
            "Install it with: pip install sae-lens"
        ) from exc

    if hook_type not in _HOOK_TYPE_MAP:
        known = ", ".join(f'"{h}"' for h in _HOOK_TYPE_MAP)
        raise ValueError(f"Unknown hook_type {hook_type!r}. Choices: {known}")

    if activation_fn not in ("topk", "relu"):
        raise ValueError(
            f"Unknown activation_fn {activation_fn!r}. Choices: \"topk\", \"relu\""
        )

    cfg = MODELS.get(config_name)
    tl_name = cfg.tl_name if cfg else config_name

    d_in = model.cfg.d_model
    hook_name = f"blocks.{layer}.{_HOOK_TYPE_MAP[hook_type]}"

    if d_sae is None:
        d_sae = d_in * expansion_factor
    else:
        expansion_factor = max(1, d_sae // d_in)

    if activation_fn == "topk":
        sae_cfg = TopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae, k=k, device=device)
    else:
        sae_cfg = StandardTrainingSAEConfig(
            d_in=d_in, d_sae=d_sae, l1_coefficient=l1_coefficient, device=device
        )

    n_steps = max(1, training_tokens // train_batch_size_tokens)
    min_batches = math.ceil(train_batch_size_tokens / context_size)
    n_batches_in_buffer = max(8, min_batches * 2)

    target_dir = os.path.join(SAES_ROOT, name)
    tmp_dir    = os.path.join(SAES_ROOT, name, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    print("Training custom SAE")
    print(f"  name:         {name}  ->  {target_dir}")
    print(f"  model:        {tl_name}")
    print(f"  hook:         {hook_name}  (layer {layer}, {hook_type})")
    print(f"  d_in -> d_sae: {d_in} -> {d_sae}  (x{expansion_factor})")
    print(f"  activation:   {activation_fn}" + (f"  k={k}" if activation_fn == "topk" else f"  l1={l1_coefficient}"))
    print(f"  lr:           {lr}")
    print(f"  steps:        {n_steps}  ({training_tokens:,} tokens ÷ {train_batch_size_tokens} batch)")
    print(f"  buffer:       {n_batches_in_buffer} batches × {context_size} ctx = {n_batches_in_buffer * context_size:,} tokens")
    print(f"  dataset:      {dataset_path}")
    print()

    runner_cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=tl_name,
        hook_name=hook_name,
        lr=lr,
        training_tokens=training_tokens,
        train_batch_size_tokens=train_batch_size_tokens,
        context_size=context_size,
        dataset_path=dataset_path,
        dataset_trust_remote_code=False,
        prepend_bos=True,
        device=device,
        checkpoint_path=tmp_dir,    # SAELens saves to {checkpoint_path}/{run_id}/final_{step}/
        save_final_checkpoint=True,
        logger=LoggingConfig(log_to_wandb=log_to_wandb),
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=32,
    )

    trained_sae = LanguageModelSAETrainingRunner(runner_cfg, override_model=model).run()

    # SAELens saves to {tmp_dir}/{run_id}/final_{step}/ — find it recursively
    # and promote the files to target_dir so `load_sae <name>` resolves cleanly.
    cfg_files = _glob.glob(os.path.join(tmp_dir, "**", "cfg.json"), recursive=True)
    if cfg_files:
        generated_dir = os.path.dirname(cfg_files[0])
        for fname in os.listdir(generated_dir):
            shutil.move(
                os.path.join(generated_dir, fname),
                os.path.join(target_dir, fname),
            )
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nTraining complete.  Load with:  load_sae {name}")
    return trained_sae

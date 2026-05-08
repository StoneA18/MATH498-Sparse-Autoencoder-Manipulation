"""
SAE steering utilities for language model experiments.

Supports GPT-2-small and Gemma-3-270m-IT with SAEs loaded from Neuronpedia.

Typical usage:

    model = load_gpt2_small()
    sae = load_sae_from_neuronpedia("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    model.add_sae(sae)

    sae.clamp(18493, 40.0)
    text = model.generate("The capital of France is")
    sae.clear()
    sae.analyze("prompt.txt", from_file=True, html_output="index.html")
"""

from __future__ import annotations

import datetime
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.distributions as tdist
from sae_lens import SAE
from transformer_lens import HookedTransformer


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _t(x: float) -> torch.Tensor:
    return torch.tensor(float(x))

_DIST_MAP = {
    "normal":      lambda p: tdist.Normal(_t(p[0]), _t(p[1])),
    "uniform":     lambda p: tdist.Uniform(_t(p[0]), _t(p[1])),
    "lognormal":   lambda p: tdist.LogNormal(_t(p[0]), _t(p[1])),
    "exponential": lambda p: tdist.Exponential(_t(p[0])),
    "beta":        lambda p: tdist.Beta(_t(p[0]), _t(p[1])),
}


class Dist:
    """
    A named probability distribution with fixed parameters.

    Supported names: normal, uniform, lognormal, exponential, beta.

    Examples:
        Dist('normal', 40, 10)      # mean=40, std=10
        Dist('uniform', 0, 1)       # low=0, high=1
        Dist('exponential', 0.5)    # rate=0.5
    """

    def __init__(self, name: str, *params: float) -> None:
        if name not in _DIST_MAP:
            raise ValueError(
                f"Unknown distribution '{name}'. Supported: {list(_DIST_MAP)}"
            )
        self.name = name
        self.params = params
        self._dist = _DIST_MAP[name](params)

    def sample(self, shape: tuple = ()) -> torch.Tensor:
        return self._dist.sample(torch.Size(shape))

    def __repr__(self) -> str:
        return f"Dist({self.name!r}, {', '.join(str(p) for p in self.params)})"


# ------------------------------------------------------------------
# Steering operations
# ------------------------------------------------------------------

class SteeringOp(ABC):
    """Base class for token-level feature steering operations.

    Subclass this and implement ``apply`` to define a new operation type.
    All built-in subclasses are importable from this module.
    """

    @abstractmethod
    def apply(self, current_val: torch.Tensor, token_idx: int) -> torch.Tensor:
        """Modify a feature activation at a given token position.

        Args:
            current_val: shape-(1,) tensor with the encoded activation value.
            token_idx:   absolute token index in the sequence (0-based, BOS = 0).
        Returns:
            Modified activation tensor of the same shape.
        """

    @property
    def referenced_features(self) -> list[int]:
        """Feature IDs this op needs to read besides the one it is applied to.

        Override in subclasses that chain behaviour from other features.
        The tracking hook calls this to know which values to collect, then
        passes them to ``set_feature_context`` before each ``apply`` call.
        """
        return []

    def set_feature_context(self, feature_values: dict) -> None:
        """Receive current activations for all ``referenced_features``.

        Called by the tracking hook immediately before ``apply``.
        ``feature_values`` maps feature_id → float (current encoded value).
        Override in any op that reads ``referenced_features``.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ClampOp(SteeringOp):
    """Always fix the feature to a constant *value*."""

    def __init__(self, value: float) -> None:
        self.value = value

    def apply(self, current_val: torch.Tensor, token_idx: int) -> torch.Tensor:
        return current_val.new_full(current_val.shape, self.value)

    def __repr__(self) -> str:
        return f"ClampOp({self.value})"


class CondDistOp(SteeringOp):
    """With probability *p*, replace the activation with a sample from *dist*."""

    def __init__(self, p: float, dist: Dist) -> None:
        self.p = p
        self.dist = dist

    def apply(self, current_val: torch.Tensor, token_idx: int) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return self.dist.sample(current_val.shape).to(current_val.device)
        return current_val

    def __repr__(self) -> str:
        return f"CondDistOp(p={self.p}, dist={self.dist!r})"


class EveryOtherTokenOp(SteeringOp):
    """Clamp to *value* on every other token.

    When *start_on=True* (default), fires on even absolute indices (0, 2, 4, …).
    """

    def __init__(self, value: float, start_on: bool = True) -> None:
        self.value = value
        self.start_on = start_on

    def apply(self, current_val: torch.Tensor, token_idx: int) -> torch.Tensor:
        if (token_idx % 2 == 0) == self.start_on:
            return current_val.new_full(current_val.shape, self.value)
        return current_val

    def __repr__(self) -> str:
        return f"EveryOtherTokenOp({self.value})"


class NthTokenOp(SteeringOp):
    """Clamp to *value* every *n*-th token (optional *offset* shifts the phase)."""

    def __init__(self, n: int, value: float, offset: int = 0) -> None:
        self.n = n
        self.value = value
        self.offset = offset

    def apply(self, current_val: torch.Tensor, token_idx: int) -> torch.Tensor:
        if (token_idx - self.offset) % self.n == 0:
            return current_val.new_full(current_val.shape, self.value)
        return current_val

    def __repr__(self) -> str:
        return f"NthTokenOp(n={self.n}, value={self.value}, offset={self.offset})"


class SpecificTokensOp(SteeringOp):
    """Clamp to *value* only at the specified absolute token indices."""

    def __init__(self, token_indices, value: float) -> None:
        self._indices: frozenset[int] = frozenset(int(i) for i in token_indices)
        self.value = value

    def apply(self, current_val: torch.Tensor, token_idx: int) -> torch.Tensor:
        if token_idx in self._indices:
            return current_val.new_full(current_val.shape, self.value)
        return current_val

    def __repr__(self) -> str:
        preview = sorted(self._indices)[:6]
        dots = "…" if len(self._indices) > 6 else ""
        return f"SpecificTokensOp({preview}{dots}, {self.value})"


class FibonacciTokensOp(SteeringOp):
    """Clamp to *value* on Fibonacci-indexed tokens (0, 1, 1, 2, 3, 5, 8, 13, …)."""

    def __init__(self, value: float, max_tokens: int = 2000) -> None:
        self.value = value
        self._fibs: frozenset[int] = self._build(max_tokens)

    @staticmethod
    def _build(max_val: int) -> frozenset:
        fibs: set[int] = {0, 1}
        a, b = 0, 1
        while b <= max_val:
            a, b = b, a + b
            fibs.add(b)
        return frozenset(fibs)

    def apply(self, current_val: torch.Tensor, token_idx: int) -> torch.Tensor:
        if token_idx in self._fibs:
            return current_val.new_full(current_val.shape, self.value)
        return current_val

    def __repr__(self) -> str:
        return f"FibonacciTokensOp({self.value})"


class ThresholdOp(SteeringOp):
    """Clamp to *value* only when the current activation exceeds *threshold*."""

    def __init__(self, threshold: float, value: float) -> None:
        self.threshold = threshold
        self.value = value

    def apply(self, current_val: torch.Tensor, token_idx: int) -> torch.Tensor:
        if current_val.item() > self.threshold:
            return current_val.new_full(current_val.shape, self.value)
        return current_val

    def __repr__(self) -> str:
        return f"ThresholdOp(threshold={self.threshold}, value={self.value})"


class ScaleOp(SteeringOp):
    """Multiply the feature activation by *scale*."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def apply(self, current_val: torch.Tensor, token_idx: int) -> torch.Tensor:
        return current_val * self.scale

    def __repr__(self) -> str:
        return f"ScaleOp({self.scale})"


class AddOp(SteeringOp):
    """Add *delta* to the feature activation (can be negative)."""

    def __init__(self, delta: float) -> None:
        self.delta = delta

    def apply(self, current_val: torch.Tensor, token_idx: int) -> torch.Tensor:
        return current_val + self.delta

    def __repr__(self) -> str:
        return f"AddOp({self.delta})"


class ZeroOp(SteeringOp):
    """Always zero out the feature activation."""

    def apply(self, current_val: torch.Tensor, token_idx: int) -> torch.Tensor:
        return torch.zeros_like(current_val)

    def __repr__(self) -> str:
        return "ZeroOp()"


class ChainOp(SteeringOp):
    """Set this feature to *ratio* × another feature's current activation.

    If the other feature is inactive (value ≤ 0), no intervention is applied
    and the feature retains its natural activation.  The other feature is never
    modified by this op.

    Example::

        # Whenever feature 18420 is active, force feature 18493 = 2 × f18420
        {18493: ChainOp(other_feature=18420, ratio=2.0)}
    """

    def __init__(self, other_feature: int, ratio: float) -> None:
        self.other_feature = other_feature
        self.ratio = ratio
        self._other_val: float = 0.0

    @property
    def referenced_features(self) -> list[int]:
        return [self.other_feature]

    def set_feature_context(self, feature_values: dict) -> None:
        self._other_val = feature_values.get(self.other_feature, 0.0)

    def apply(self, current_val: torch.Tensor, token_idx: int) -> torch.Tensor:
        if self._other_val > 0:
            return current_val.new_full(current_val.shape, self.ratio * self._other_val)
        return current_val

    def __repr__(self) -> str:
        return f"ChainOp(other_feature={self.other_feature}, ratio={self.ratio})"


class SteerableSAE:
    """
    Wraps a pretrained SAE with feature-clamping and token-level analysis.

    Must be attached to a SteerableModel via model.add_sae(sae) before
    calling analyze().
    """

    def __init__(self, sae: SAE, hook_name: str, neuronpedia_id: Optional[str] = None) -> None:
        self._sae = sae.to(_DEVICE)
        self.hook_name = hook_name
        self.neuronpedia_id = neuronpedia_id
        self._clamps: dict[int, float] = {}
        self._cond_dists: dict[int, tuple[float, Dist]] = {}
        self._model: Optional[SteerableModel] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clamp(self, feature_id: int, value: float) -> None:
        """Fix feature_id to value before decoding on every forward pass."""
        self._clamps[feature_id] = value

    def cond_dist(self, feature_id: int, p: float, dist: Dist) -> None:
        """
        With probability p, replace feature_id's activation with a sample from dist.
        Applied independently to each token position on every forward pass.
        """
        self._cond_dists[feature_id] = (p, dist)

    def clear(self) -> None:
        """Remove all active clamps and conditional distributions."""
        self._clamps.clear()
        self._cond_dists.clear()

    def analyze(
        self,
        prompt: str,
        from_file: bool = False,
        html_output: Optional[str] = None,
        top_n: int = 10,
    ) -> str:
        """
        Run the SAE over the prompt and generate an HTML analysis page.

        The page shows a ranked feature table per token (matching the
        example_index.html design) and a click-to-expand detail panel.

        Returns the HTML string and, if html_output is given, also writes it.
        """
        if self._model is None:
            raise RuntimeError(
                "Attach this SAE to a SteerableModel via model.add_sae() before calling analyze()."
            )

        if from_file:
            prompt = Path(prompt).read_text(encoding="utf-8").strip()

        tl_model = self._model._model
        str_tokens: list[str] = tl_model.to_str_tokens(prompt)  # type: ignore[assignment]
        feature_matrix = self._forward_collect(tl_model, prompt)

        model_name = getattr(tl_model.cfg, "model_name", "")
        html = _build_analysis_html(
            str_tokens=str_tokens,
            feature_matrix=feature_matrix,
            hook_name=self.hook_name,
            model_name=model_name,
            top_n=top_n,
        )
        if html_output:
            Path(html_output).write_text(html, encoding="utf-8")
        return html

    def collect_activations(
        self,
        text: str,
        from_file: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Run text through the SAE and return raw feature activations.

        Long texts are split into chunks so the model's context window is
        never exceeded.  chunk_size defaults to n_ctx - 2.

        Returns a tensor of shape (n_tokens, n_features).
        """
        if self._model is None:
            raise RuntimeError(
                "Attach this SAE to a SteerableModel via model.add_sae() before calling collect_activations()."
            )

        if from_file:
            text = Path(text).read_text(encoding="utf-8").strip()

        tl_model = self._model._model
        ctx = int(tl_model.cfg.n_ctx)
        cs = chunk_size or (ctx - 2)

        # tokenise without BOS so we can slice freely
        all_tokens = tl_model.to_tokens(text, prepend_bos=False)[0]  # (N,)

        chunks: list[torch.Tensor] = []
        for start in range(0, len(all_tokens), cs):
            chunk = all_tokens[start : start + cs].unsqueeze(0)  # (1, L)
            chunks.append(self._forward_collect(tl_model, chunk))

        return torch.cat(chunks, dim=0)  # (total_tokens, n_features)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_collect(
        self,
        tl_model,
        input,  # str or (1, L) token tensor
    ) -> torch.Tensor:
        """Single forward pass; returns feature_matrix (seq_len, n_features)."""
        storage: list[torch.Tensor] = []

        def _hook(value: torch.Tensor, hook=None):
            with torch.no_grad():
                storage.append(self._sae.encode(value).cpu().float())
            return value

        with tl_model.hooks(fwd_hooks=[(self.hook_name, _hook)]):
            tl_model(input)

        if not storage:
            raise RuntimeError(
                f"Hook '{self.hook_name}' never fired. Check the hook name."
            )
        return storage[0][0]  # (seq_len, n_features)

    # ------------------------------------------------------------------
    # Internal hook factories
    # ------------------------------------------------------------------

    def _generation_hook(self):
        clamps = dict(self._clamps)
        cond_dists = dict(self._cond_dists)
        sae = self._sae

        def _hook(value: torch.Tensor, hook=None):
            with torch.no_grad():
                acts = sae.encode(value)
                for fid, val in clamps.items():
                    acts[..., fid] = val
                for fid, (p, dist) in cond_dists.items():
                    shape = acts.shape[:-1]  # (batch, seq_len)
                    mask = torch.rand(shape, device=acts.device) < p
                    samples = dist.sample(shape).to(acts.device)
                    acts[..., fid] = torch.where(mask, samples, acts[..., fid])
                return sae.decode(acts)

        return _hook

    def _make_tracking_hook(
        self,
        ops: dict,       # {feature_id: SteeringOp}
        track_fids: list,
    ) -> tuple:
        """Create a hook that applies per-feature ops and records activation values.

        Compatible with TransformerLens KV-cache generation: tracks an absolute
        position counter so it works whether seq_len is 1 (KV-cache) or growing.

        Returns ``(hook_fn, log)`` where ``log`` is populated during generation.
        After generation, ``log[i]`` is ``{feature_id: {"v": post_op, "n": natural}}``
        for token i, where ``v`` is the intervention-modified value and ``n`` is
        the natural (pre-op) activation. Equal values mean the op did not fire.
        """
        log: list[dict] = []
        abs_pos = [0]
        sae = self._sae

        def _hook(value: torch.Tensor, hook=None):
            with torch.no_grad():
                acts = sae.encode(value)         # (batch, seq_len, n_features)
                _, seq_len, _ = acts.shape

                for local_pos in range(seq_len):
                    cur_abs = abs_pos[0]

                    # Snapshot natural values before any op fires
                    natural = {fid: acts[0, local_pos, fid].item() for fid in track_fids}

                    for fid, op in ops.items():
                        ref = op.referenced_features
                        if ref:
                            op.set_feature_context(
                                {rfid: acts[0, local_pos, rfid].item() for rfid in ref}
                            )
                        cur_val = acts[0, local_pos, fid : fid + 1]
                        acts[0, local_pos, fid] = op.apply(cur_val, cur_abs).item()

                    # Store both post-op value and natural value so the dashboard
                    # can distinguish "intervention fired" vs "naturally active".
                    log.append({
                        fid: {"v": acts[0, local_pos, fid].item(), "n": natural[fid]}
                        for fid in track_fids
                    })
                    abs_pos[0] += 1

                return sae.decode(acts)

        return _hook, log


class SteerableModel:
    """
    Wraps a TransformerLens HookedTransformer with zero or more attached SAEs.

    All attached SAEs (with their current clamps) are applied during generate().
    """

    def __init__(self, model: HookedTransformer) -> None:
        self._model = model
        self._saes: list[SteerableSAE] = []

    def add_sae(self, sae: SteerableSAE) -> SteerableSAE:
        """Attach a SteerableSAE. Returns the sae for convenience."""
        sae._model = self
        self._saes.append(sae)
        url = (
            f"https://www.neuronpedia.org/{sae.neuronpedia_id}"
            if sae.neuronpedia_id
            else None
        )
        msg = f"SAE attached at hook '{sae.hook_name}'."
        if url:
            msg += f" Browse features: {url}"
        print(msg)
        return sae

    def cond_dist(self, feature_id: int, p: float, dist: Dist) -> None:
        """
        Apply a conditional distribution to feature_id on all attached SAEs.
        With probability p, replace the activation with a sample from dist;
        otherwise leave it unchanged.
        """
        if not self._saes:
            raise RuntimeError("No SAEs attached. Call add_sae() first.")
        for sae in self._saes:
            sae.cond_dist(feature_id, p, dist)

    def chat(self, max_tokens: int = 200) -> None:
        """
        Interactive chat loop: type a prompt, see the model's completion.
        Type 'quit' (or press Ctrl-C) to exit.
        All currently active SAE clamps are applied on every turn.
        """
        print("Chatting with the model. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.lower() == "quit":
                break
            if not prompt:
                continue
            result = self.generate(prompt, max_tokens=max_tokens)
            # strip the echoed prompt so only the completion is shown
            completion = result[len(prompt):]
            print(completion.strip(), "\n")

    def generate(
        self,
        prompt: str,
        from_file: bool = False,
        output_file: Optional[str] = None,
        max_tokens: int = 100,
        log: bool = True,
    ) -> str:
        """
        Generate text with all active SAE clamps applied.

        Returns the full text (prompt + generation). If output_file is given,
        also writes the result there. Unless log=False, appends a full entry
        to output/log.txt.
        """
        if from_file:
            prompt = Path(prompt).read_text(encoding="utf-8").strip()

        for sae in self._saes:
            self._model.add_hook(sae.hook_name, sae._generation_hook())

        try:
            result: str = self._model.generate(
                prompt,
                max_new_tokens=max_tokens,
                verbose=False,
            )
        finally:
            self._model.reset_hooks()

        if output_file:
            Path(output_file).write_text(result, encoding="utf-8")

        if log:
            self._log_generation(prompt, result, max_tokens)

        return result

    def _log_generation(self, prompt: str, result: str, max_tokens: int) -> None:
        log_path = Path("output/log.txt")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        model_name = getattr(self._model.cfg, "model_name", str(self._model.cfg))
        completion = result[len(prompt):]
        n_tokens = len(self._model.to_tokens(completion, prepend_bos=False)[0])

        sae_entries = []
        for sae in self._saes:
            clamps = {str(fid): val for fid, val in sae._clamps.items()}
            cond_dists = {
                str(fid): {"p": p, "dist": d.name, "params": list(d.params)}
                for fid, (p, d) in sae._cond_dists.items()
            }
            sae_entries.append({
                "hook_name": sae.hook_name,
                "neuronpedia_id": sae.neuronpedia_id,
                "clamps": clamps,
                "cond_dists": cond_dists,
            })

        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "max_tokens": max_tokens,
            "n_tokens_generated": n_tokens,
            "saes": sae_entries,
            "prompt": prompt,
            "response": completion.strip(),
        }

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------
# Factory functions
# ------------------------------------------------------------------

def load_gpt2_small() -> SteerableModel:
    """Load GPT-2-small via TransformerLens."""
    model = HookedTransformer.from_pretrained("gpt2", device=_DEVICE)
    return SteerableModel(model)


def load_gemma_3_270m_it() -> SteerableModel:
    """Load Gemma-3-270m-IT via TransformerLens."""
    model = HookedTransformer.from_pretrained(
        "google/gemma-3-270m-it",
        device=_DEVICE,
    )
    return SteerableModel(model)


def load_sae_from_neuronpedia(
    release: str,
    sae_id: str,
    *,
    hook_name: Optional[str] = None,
) -> SteerableSAE:
    """
    Load a pretrained SAE from Neuronpedia.

    Args:
        release:   Neuronpedia release string, e.g. "gpt2-small-res-jb".
        sae_id:    SAE ID within the release, e.g. "blocks.8.hook_resid_pre".
        hook_name: TransformerLens hook point to intercept. Defaults to sae_id.

    Example:
        sae = load_sae_from_neuronpedia("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    """
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=_DEVICE)
    np_id: Optional[str] = sae.cfg.metadata.get("neuronpedia_id")
    resolved_hook = hook_name or sae.cfg.metadata.hook_name or sae_id
    return SteerableSAE(sae, hook_name=resolved_hook, neuronpedia_id=np_id)


# ------------------------------------------------------------------
# HTML generation
# ------------------------------------------------------------------

def _build_analysis_html(
    str_tokens: list[str],
    feature_matrix: torch.Tensor,  # (seq_len, n_features)
    hook_name: str = "",
    model_name: str = "",
    top_n: int = 10,
) -> str:
    seq_len = feature_matrix.shape[0]

    # top_features[t] = [[fid, val], ...] sorted by val desc, capped at top_n
    top_features_data: list[list] = []
    for t in range(seq_len):
        acts = feature_matrix[t]
        nz_idx = torch.where(acts != 0)[0].tolist()
        nz_val = [float(acts[i]) for i in nz_idx]
        pairs = sorted(zip(nz_idx, nz_val), key=lambda x: -x[1])
        top_features_data.append([[int(f), round(v, 6)] for f, v in pairs[:top_n]])

    # dense activations for every feature that appears in any top-n slot
    unique_fids: set[int] = {fid for tf in top_features_data for fid, _ in tf}
    feature_activations: dict[str, list[float]] = {
        str(fid): [round(float(feature_matrix[t, fid].item()), 6) for t in range(seq_len)]
        for fid in unique_fids
    }

    data_json = json.dumps(
        {
            "tokens": list(str_tokens),
            "topFeatures": top_features_data,
            "featureActivations": feature_activations,
            "topN": top_n,
            "modelName": model_name,
            "hookPoint": hook_name,
        },
        ensure_ascii=False,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SAE Feature Analysis</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 16px 20px;
         background: #f5f6f8; color: #1a1a2e; }}
  h1 {{ margin: 0 0 4px; font-size: 1.15rem; font-weight: 600; }}
  .meta {{ color: #6c757d; font-size: 0.82rem; margin-bottom: 14px; font-family: monospace; }}
  .search-bar {{ display: flex; gap: 8px; align-items: center; margin-bottom: 14px; flex-wrap: wrap; }}
  .search-bar label {{ font-size: 0.88rem; font-weight: 500; }}
  .search-bar input {{ padding: 5px 9px; border: 1px solid #ced4da; border-radius: 4px;
                      font-size: 0.88rem; width: 140px; }}
  .search-bar input:focus {{ outline: 2px solid #4361ee; border-color: transparent; }}
  .search-bar button {{ padding: 5px 13px; background: #4361ee; color: #fff;
                       border: none; border-radius: 4px; cursor: pointer; font-size: 0.88rem; }}
  .search-bar button:hover {{ background: #3451d1; }}
  .err {{ color: #d62839; font-size: 0.82rem; }}
  #detail {{ display: none; background: #fff; border: 1px solid #dee2e6; border-radius: 6px;
            padding: 14px 16px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
  #detail h2 {{ margin: 0 0 10px; font-size: 0.95rem; }}
  #detail .close-btn {{ float: right; background: none; border: none; cursor: pointer;
                       font-size: 1rem; color: #6c757d; line-height: 1; }}
  #detail .close-btn:hover {{ color: #1a1a2e; }}
  #dtable {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; }}
  #dtable th {{ background: #f1f3f5; padding: 5px 10px; text-align: left;
               border-bottom: 2px solid #dee2e6; font-weight: 600; }}
  #dtable td {{ padding: 3px 10px; border-bottom: 1px solid #f1f3f5; font-family: monospace; }}
  #dtable tr.zero td {{ color: #adb5bd; }}
  .bar-wrap {{ width: 140px; }}
  .bar {{ display: inline-block; height: 11px; background: #4361ee; border-radius: 2px;
         vertical-align: middle; min-width: 0; }}
  #main-wrap {{ overflow-x: auto; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
  #main {{ border-collapse: collapse; white-space: nowrap; font-size: 0.8rem;
          background: #fff; width: 100%; }}
  #main thead th {{ background: #1a1a2e; color: #e9ecef; padding: 6px 10px; text-align: left;
                   position: sticky; top: 0; z-index: 2; }}
  #main thead th.pos-h {{ width: 40px; }}
  #main thead th.tok-h {{ min-width: 80px; }}
  #main td {{ border-bottom: 1px solid #e9ecef; padding: 4px 6px; vertical-align: top; }}
  #main td.pos {{ color: #adb5bd; font-size: 0.75rem; text-align: right; width: 36px;
                 background: #f8f9fa; position: sticky; left: 0; z-index: 1; }}
  #main td.tok {{ font-family: monospace; font-weight: 600; background: #f8f9fa;
                 position: sticky; left: 36px; z-index: 1; padding: 4px 10px; min-width: 70px; }}
  .fc {{ display: inline-block; padding: 2px 7px; border-radius: 3px; cursor: pointer;
        font-family: monospace; line-height: 1.5; transition: filter .1s; }}
  .fc:hover {{ filter: brightness(.88); }}
  .fc.active {{ outline: 2px solid #1a1a2e; outline-offset: 1px; }}
  .fc small {{ opacity: .75; font-size: .9em; }}
</style>
</head>
<body>
<h1>SAE Feature Analysis</h1>
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
const D = {data_json};

const fMax = {{}};
for (const [k, vs] of Object.entries(D.featureActivations))
  fMax[k] = Math.max(0, ...vs);

function heatRgb(val, maxVal) {{
  if (!maxVal) return '#f8f9fa';
  const t = Math.min(val / maxVal, 1);
  const r = Math.round(255 + t * (67  - 255));
  const g = Math.round(255 + t * (97  - 255));
  const b = Math.round(255 + t * (238 - 255));
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function textColor(val, maxVal) {{
  return (maxVal && val / maxVal > 0.55) ? '#fff' : '#1a1a2e';
}}

function escHtml(s) {{
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

let hh = '<tr><th class="pos-h">#</th><th class="tok-h">Token</th>';
for (let r = 1; r <= D.topN; r++) hh += `<th>Rank&nbsp;${{r}}</th>`;
hh += '</tr>';
document.getElementById('mhead').innerHTML = hh;

let bh = '';
for (let i = 0; i < D.tokens.length; i++) {{
  bh += `<tr><td class="pos">${{i}}</td><td class="tok">${{escHtml(D.tokens[i])}}</td>`;
  for (let r = 0; r < D.topN; r++) {{
    const entry = D.topFeatures[i][r];
    if (!entry) {{ bh += '<td></td>'; continue; }}
    const [fid, val] = entry;
    const mx = fMax[String(fid)] || 1;
    const bg = heatRgb(val, mx);
    const fg = textColor(val, mx);
    bh += `<td><span class="fc" style="background:${{bg}};color:${{fg}}"
      data-fid="${{fid}}" onclick="showFeature(${{fid}})"
      title="feature ${{fid}} = ${{val.toFixed(4)}}">${{fid}} <small>(${{val.toFixed(2)}})</small></span></td>`;
  }}
  bh += '</tr>';
}}
document.getElementById('mbody').innerHTML = bh;

document.getElementById('meta').textContent =
  `model: ${{D.modelName}}  |  hook: ${{D.hookPoint}}  |  ` +
  `${{D.tokens.length}} tokens  |  ${{Object.keys(D.featureActivations).length}} unique features in top ${{D.topN}}`;

let activeId = null;
function showFeature(fid) {{
  const key = String(fid);
  if (!D.featureActivations[key]) {{
    document.getElementById('serr').textContent =
      `Feature ${{fid}} never reached top ${{D.topN}} — not included in this report.`;
    return;
  }}
  document.getElementById('serr').textContent = '';
  activeId = fid;
  document.querySelectorAll('.fc').forEach(el =>
    el.classList.toggle('active', Number(el.dataset.fid) === fid));
  document.getElementById('did').textContent = fid;
  const vals = D.featureActivations[key];
  const mx = fMax[key] || 1;
  const BAR = 130;
  let rows = '';
  for (let i = 0; i < D.tokens.length; i++) {{
    const v = vals[i];
    const w = Math.round((v / mx) * BAR);
    rows += `<tr class="${{v === 0 ? 'zero' : ''}}">
      <td>${{i}}</td><td>${{escHtml(D.tokens[i])}}</td>
      <td>${{v.toFixed(4)}}</td>
      <td class="bar-wrap"><span class="bar" style="width:${{w}}px"></span></td>
    </tr>`;
  }}
  document.getElementById('dbody').innerHTML = rows;
  const panel = document.getElementById('detail');
  panel.style.display = 'block';
  panel.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
}}

function closeDetail() {{
  document.getElementById('detail').style.display = 'none';
  document.querySelectorAll('.fc.active').forEach(el => el.classList.remove('active'));
  activeId = null;
}}

function doSearch() {{
  const v = parseInt(document.getElementById('fsearch').value, 10);
  if (isNaN(v)) {{ document.getElementById('serr').textContent = 'Enter an integer feature ID.'; return; }}
  showFeature(v);
}}

document.getElementById('fsearch').addEventListener('keydown', e => {{
  if (e.key === 'Enter') doSearch();
}});
</script>
</body>
</html>"""

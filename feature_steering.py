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
    sae_release:        Optional[str]   = None   # None = no SAE; feature commands disabled
    hook_point:         Optional[str]   = None
    neuronpedia_id:     Optional[str]   = None

    @property
    def has_sae(self) -> bool:
        return self.sae_release is not None


MODELS: Dict[str, ModelConfig] = {
    # ── GPT-2 Small ──────────────────────────────────────────────────────────
    # Tiny raw language model (not instruction-tuned). Useful for understanding
    # the steering mechanics at low cost, but cannot reliably answer questions.
    "gpt2-small": ModelConfig(
        display_name   = "GPT-2 Small (124M, raw LM)",
        tl_name        = "gpt2",
        sae_release    = "gpt2-small-res-jb",
        hook_point     = "blocks.8.hook_resid_pre",   # layer 8 of 12
        neuronpedia_id = "gpt2-small/8-res-jb",
        tl_kwargs      = dict(
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            refactor_factored_attn_matrices=True,
        ),
        is_chat       = False,
        chat_template = None,
    ),

    # ── Gemma 2B Instruction-Tuned ───────────────────────────────────────────
    # A proper instruction-tuned model that can answer questions, follow
    # directions, and hold a conversation. The SAE was trained by Joseph Bloom
    # on layer 12's residual stream (post-MLP), giving 16,384 features.
    # First load will download ~5 GB of weights — subsequent loads are cached.
    "gemma-2b-it": ModelConfig(
        display_name   = "Gemma 2B Instruction-Tuned (2B, chat model)",
        tl_name        = "gemma-2b-it",
        sae_release    = "gemma-2b-it-res-jb",
        hook_point     = "blocks.12.hook_resid_post",  # layer 12 of 18
        neuronpedia_id = "gemma-2b/12-res-jb",
        # bfloat16: same memory as float16 (~4 GB) but natively supported on
        # modern CPUs (AVX512-BF16) so much faster than float16 on CPU.
        # float16 is a GPU format — on CPU PyTorch falls back to float32 ops
        # internally anyway, making it slower than just using bfloat16 directly.
        tl_kwargs           = dict(dtype=torch.bfloat16),
        is_chat             = True,
        # Standard Gemma IT chat format. The model generates after <start_of_turn>model.
        chat_template       = (
            "<start_of_turn>user\n"
            "{prompt}<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
        # Keep responses short on CPU — a 2B model does ~1-3 tokens/sec on CPU.
        # Use `generate ... -n 200` to override when you want a longer answer.
        default_max_tokens  = 50,
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

    print(f"Loading SAE ({config.hook_point})...")
    # sae_lens >= 6.x: from_pretrained returns just the SAE;
    # from_pretrained_with_cfg_and_sparsity returns (sae, cfg_dict, sparsity).
    sae, cfg, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=config.sae_release,
        sae_id=config.hook_point,
        device=device,
    )
    sae.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model:  {config.display_name} — {n_params:,} parameters")
    print(f"  SAE:    {cfg['d_in']} residual dims -> {cfg['d_sae']} features")
    print(f"  Hook:   {config.hook_point}")
    print()
    return model, sae, config


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
        # Maps feature_id → clamped_value. Empty dict = no steering.
        self.clamped_features: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Managing clamps
    # ------------------------------------------------------------------

    def clamp(self, feature_id: int, value: float = 20.0):
        """Force feature_id to always equal value during generation."""
        self.clamped_features[feature_id] = value

    def unclamp(self, feature_id: int):
        """Remove the clamp on a single feature (returns to normal)."""
        self.clamped_features.pop(feature_id, None)

    def unclamp_all(self):
        """Remove all feature clamps — model returns to normal behavior."""
        self.clamped_features.clear()

    def list_clamps(self) -> Dict[int, float]:
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

                # Step 4: Overwrite the clamped features at every token position.
                for feat_id, feat_val in clamped.items():
                    feature_acts[:, :, feat_id] = feat_val

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

        # Register our hook persistently so it fires during every internal
        # forward pass that .generate() makes (one per new token, with KV cache).
        try:
            self.model.add_hook(self.hook_point, self._build_hook())
            with torch.no_grad():
                output = self.model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    verbose=False,   # suppress TransformerLens's own progress bar
                )
        finally:
            # Always clean up hooks, even on error.
            self.model.reset_hooks()

        # Decode just the newly generated tokens (not the prompt).
        prompt_len = tokens.shape[1]
        new_tokens = output[0, prompt_len:]
        return self.model.to_string(new_tokens)

"""
Configurable, trainable Sparse Autoencoders for TransformerLens models.

This module is intentionally small and hackable.  The main pieces are:

  - SAEConfig: defines the SAE structure and activation.
  - AbstractSAE: interface for custom SAE implementations.
  - TrainableSAE: a default linear encoder/decoder SAE you can subclass.
  - SAEConnector: attaches an SAE to any TransformerLens hook point.
  - load_hooked_transformer: convenience loader matching this repo's packages.

Example
-------
    device = resolve_device("auto")
    model = load_hooked_transformer("gpt2", device=device)
    cfg = SAEConfig(d_in=model.cfg.d_model, d_sae=8192, activation="topk", k=32)
    sae = TrainableSAE(cfg, device=device)

    connector = SAEConnector(
        model=model,
        sae=sae,
        hook_point=HookPointSpec(layer=8, site="resid_pre").name,
    )

    # Train from model activations.
    fit_sae_on_texts(sae, connector, ["Hello world"], steps=10)

    # Run the model with the SAE inserted as a reconstruction hook.
    logits = connector.run_with_sae(model.to_tokens("Hello world"))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import importlib.metadata
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
ActivationLike = Union[str, Callable[[torch.Tensor], torch.Tensor], nn.Module]
FeatureProjector = Callable[[torch.Tensor], torch.Tensor]
ProjectorLocation = Literal["pre_activation", "post_activation"]


def _patch_torch_metadata_version() -> None:
    """
    Work around local envs where importlib metadata returns None for torch.

    Some versions of datasets/transformer_lens parse importlib.metadata.version("torch")
    during import. In this venv that value can be None, while torch.__version__ is valid.
    """
    if getattr(importlib.metadata, "_trainable_sae_torch_version_patch", False):
        return

    metadata_version = importlib.metadata.version

    def version_with_torch_fallback(package_name: str) -> str:
        version = metadata_version(package_name)
        if package_name == "torch" and version is None:
            return torch.__version__.split("+", maxsplit=1)[0]
        return version

    importlib.metadata.version = version_with_torch_fallback
    importlib.metadata._trainable_sae_torch_version_patch = True


def _exception_chain_text(exc: BaseException) -> str:
    """Collect chained exception text so wrapped HF errors remain diagnosable."""
    messages = []
    current: Optional[BaseException] = exc
    while current is not None:
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    return "\n".join(messages)


def resolve_device(device: Optional[str] = "auto") -> str:
    """
    Resolve a user-specified device string.

    Args:
        device: "auto", "cpu", "cuda", "cuda:0", "mps", etc.

    Returns:
        A torch-compatible device string. "auto" picks CUDA when available,
        then Apple MPS when available, otherwise CPU.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device {device!r}, but CUDA is not available.")
    if device.startswith("mps"):
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            raise ValueError(f"Requested device {device!r}, but MPS is not available.")
    return device


@dataclass
class HookPointSpec:
    """
    Human-friendly way to choose where the SAE attaches in a TransformerLens model.

    Common sites:
      resid_pre, resid_post, mlp_out, attn_out

    You can also skip this class and pass a raw hook name such as
    "blocks.8.hook_resid_pre" directly to SAEConnector.
    """

    layer: int
    site: str = "resid_post"

    SITE_TO_HOOK: Dict[str, str] = field(
        default_factory=lambda: {
            "resid_pre": "hook_resid_pre",
            "resid_post": "hook_resid_post",
            "mlp_out": "hook_mlp_out",
            "attn_out": "hook_attn_out",
        }
    )

    @property
    def name(self) -> str:
        if self.site.startswith("hook_"):
            hook_site = self.site
        else:
            try:
                hook_site = self.SITE_TO_HOOK[self.site]
            except KeyError as exc:
                known = ", ".join(sorted(self.SITE_TO_HOOK))
                raise ValueError(f"Unknown hook site {self.site!r}. Known: {known}") from exc
        return f"blocks.{self.layer}.{hook_site}"


@dataclass
class SAEConfig:
    """Structure and training defaults for TrainableSAE."""

    d_in: int
    d_sae: int
    activation: str = "relu"
    k: int = 50
    shrink_threshold: float = 1.0
    pre_layer_norm: bool = False
    bias: bool = True
    tied_decoder: bool = False
    normalize_decoder: bool = True
    l1_coefficient: float = 1e-4
    l1_context_coef: float = 0.0
    lr: float = 2e-4
    dtype: str = "float32"
    device: str = "auto"
    metadata: Dict[str, object] = field(default_factory=dict)


class TopKActivation(nn.Module):
    """Keep the k largest positive pre-activations and zero everything else."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, indices = x.topk(min(self.k, x.shape[-1]), dim=-1)
        values = F.relu(values)
        out = torch.zeros_like(x)
        return out.scatter(-1, indices, values)


class TopBottomKActivation(nn.Module):
    """Keep the k largest-magnitude pre-activations, preserving their signs."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = x.abs().topk(min(self.k, x.shape[-1]), dim=-1).indices
        out = torch.zeros_like(x)
        return out.scatter(-1, indices, x.gather(-1, indices))


class ShrinkActivation(nn.Module):
    """Soft-threshold activations, preserving signs while shrinking magnitudes."""

    def __init__(self, threshold: float = 1.0):
        super().__init__()
        if threshold < 0:
            raise ValueError("shrink threshold must be non-negative.")
        self.threshold = float(threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sign() * F.relu(x.abs() - self.threshold)


def build_activation(
    activation: ActivationLike,
    k: int = 50,
    shrink_threshold: float = 1.0,
) -> nn.Module:
    """Turn a string/callable/module into an nn.Module activation."""
    if isinstance(activation, nn.Module):
        return activation
    if callable(activation):
        return _CallableActivation(activation)

    name = activation.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    if name == "topk":
        return TopKActivation(k)
    if name in ("tbk", "topbottomk", "top_bottom_k"):
        return TopBottomKActivation(k)
    if name in ("shrink", "softshrink", "soft_shrink", "soft_threshold"):
        return ShrinkActivation(shrink_threshold)
    raise ValueError(
        f"Unknown activation {activation!r}. "
        "Use relu, gelu, sigmoid, tanh, identity, topk, tbk, shrink, or a callable/module."
    )


class _CallableActivation(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class AbstractSAE(nn.Module, ABC):
    """
    Interface for SAEs that can be trained and inserted into a TransformerLens hook.

    Subclass this when you want a different structure, e.g. a gated SAE,
    convolutional encoder, deeper MLP encoder, custom normalization, etc.
    """

    @property
    @abstractmethod
    def d_in(self) -> int:
        """Activation-space dimension consumed by the SAE."""

    @property
    @abstractmethod
    def d_sae(self) -> int:
        """Feature-space dimension produced by the SAE."""

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map model activations [..., d_in] to sparse features [..., d_sae]."""

    @abstractmethod
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Map sparse features [..., d_sae] back to activations [..., d_in]."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return self.decode(features), features

    def loss(
        self,
        x: torch.Tensor,
        recon: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Default MSE + L1 loss. Override for custom training objectives."""
        if recon is None or features is None:
            recon, features = self(x)

        if loss_mask is None:
            mse = F.mse_loss(recon, x)
        else:
            mask = loss_mask.to(device=x.device, dtype=x.dtype)
            while mask.ndim < x.ndim:
                mask = mask.unsqueeze(-1)
            valid_tokens = mask.sum().clamp_min(1.0)
            mse = ((recon - x).pow(2) * mask).sum() / (valid_tokens * x.shape[-1])

        regularization, metrics = self.regularization_loss(features, loss_mask=loss_mask)
        total = mse + regularization
        metrics = {
            "loss": float(total.detach()),
            "mse": float(mse.detach()),
            **metrics,
        }
        return total, metrics

    def regularization_loss(
        self,
        features: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return the SAE L1 penalties used by all training objectives."""
        if loss_mask is None:
            l1 = features.abs().mean()
        else:
            mask = loss_mask.to(device=features.device, dtype=features.dtype)
            while mask.ndim < features.ndim:
                mask = mask.unsqueeze(-1)
            valid_tokens = mask.sum().clamp_min(1.0)
            l1 = (features.abs() * mask).sum() / (valid_tokens * features.shape[-1])

        l1_coeff = float(getattr(getattr(self, "cfg", None), "l1_coefficient", 0.0))
        l1_context_coef = float(getattr(getattr(self, "cfg", None), "l1_context_coef", 0.0))
        context_l1 = features.new_tensor(0.0)
        if l1_context_coef != 0.0 and features.ndim >= 3:
            feature_abs = features.abs()
            if loss_mask is not None:
                context_mask = loss_mask.to(device=features.device, dtype=features.dtype)
                while context_mask.ndim < features.ndim:
                    context_mask = context_mask.unsqueeze(-1)
                feature_abs = feature_abs * context_mask
                valid_per_context = loss_mask.to(device=features.device, dtype=features.dtype).sum(dim=-1).clamp_min(1.0)
            else:
                valid_per_context = features.new_full(features.shape[:-2], features.shape[-2])

            # Sum each feature's L1 mass across the token dimension, then square
            # it so reusing one feature on many tokens is costlier than spreading
            # that mass across distinct features.
            context_feature_mass = feature_abs.sum(dim=-2)
            context_l1 = (context_feature_mass.pow(2) / valid_per_context.unsqueeze(-1)).mean()

        total = l1_coeff * l1 + l1_context_coef * context_l1
        metrics = {
            "l1": float(l1.detach()),
        }
        if l1_context_coef != 0.0:
            metrics["l1_context"] = float(context_l1.detach())
            metrics["l1_context_coef"] = l1_context_coef
        return total, metrics


class TrainableSAE(AbstractSAE):
    """Default linear encoder/decoder SAE with configurable activation."""

    def __init__(
        self,
        cfg: SAEConfig,
        activation: Optional[ActivationLike] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)
        resolved_device = resolve_device(device or cfg.device)
        self.cfg.device = resolved_device

        self.encoder = nn.Linear(
            cfg.d_in, cfg.d_sae, bias=cfg.bias, dtype=dtype, device=resolved_device
        )
        self.decoder = nn.Linear(
            cfg.d_sae, cfg.d_in, bias=cfg.bias, dtype=dtype, device=resolved_device
        )
        self.activation_fn = build_activation(
            activation or cfg.activation,
            k=cfg.k,
            shrink_threshold=cfg.shrink_threshold,
        )
        self.pre_layer_norm = (
            nn.LayerNorm(cfg.d_in, elementwise_affine=False, dtype=dtype, device=resolved_device)
            if cfg.pre_layer_norm
            else nn.Identity()
        )
        self.reset_parameters()

    @property
    def d_in(self) -> int:
        return self.cfg.d_in

    @property
    def d_sae(self) -> int:
        return self.cfg.d_sae

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)
        self.normalize_decoder_weights()

    def normalize_decoder_weights(self) -> None:
        if self.cfg.normalize_decoder:
            with torch.no_grad():
                self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_layer_norm(x)
        return self.encoder(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(self.pre_activations(x))

    def encode_with_projector(
        self,
        x: torch.Tensor,
        sae_projector: FeatureProjector,
    ) -> torch.Tensor:
        pre_activations = self.pre_activations(x)
        projected = sae_projector(pre_activations)
        if projected.shape != pre_activations.shape:
            raise ValueError(
                "sae_projector must return a tensor with the same shape as its input: "
                f"got {tuple(projected.shape)}, expected {tuple(pre_activations.shape)}."
            )
        return self.activation_fn(projected)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        if self.cfg.tied_decoder:
            bias = self.decoder.bias if self.decoder.bias is not None else None
            return F.linear(features, self.encoder.weight.T, bias)
        return self.decoder(features)

    def training_step(
        self,
        activations: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad_norm: Optional[float] = 1.0,
        record_nonzero_features: bool = False,
        sparsity_udf: Optional[Callable[["TrainableSAE", Dict[str, float]], None]] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        recon, features = self(activations)
        loss, metrics = self.loss(activations, recon, features, loss_mask=loss_mask)
        if record_nonzero_features or sparsity_udf is not None:
            nonzero_features = (features != 0).sum(dim=-1).float()
            if loss_mask is None:
                metrics["avg_nonzero_features"] = float(nonzero_features.mean().detach())
            else:
                mask = loss_mask.to(device=features.device, dtype=nonzero_features.dtype)
                metrics["avg_nonzero_features"] = float(
                    (nonzero_features * mask).sum().div(mask.sum().clamp_min(1.0)).detach()
                )
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
        optimizer.step()
        self.normalize_decoder_weights()
        if sparsity_udf is not None:
            sparsity_udf(self, metrics)
        return metrics

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"cfg": asdict(self.cfg), "state_dict": self.state_dict()},
            path / "trainable_sae.pt",
        )

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "TrainableSAE":
        resolved_device = resolve_device(device)
        checkpoint = torch.load(Path(path) / "trainable_sae.pt", map_location=resolved_device)
        cfg_dict = dict(checkpoint["cfg"])
        cfg_dict["device"] = resolved_device
        sae = cls(SAEConfig(**cfg_dict), device=resolved_device)
        sae.load_state_dict(checkpoint["state_dict"])
        return sae.to(resolved_device)


class SAEConnector:
    """
    Connect an AbstractSAE to a HookedTransformer hook point.

    Modes:
      reconstruct: replace activations with the SAE reconstruction
      cache:       do not modify activations; just keep latest features/reconstruction

    If preserve_error=True, the hook adds back the original SAE reconstruction
    error after any feature_transform is applied.  This matches the style used
    in feature_steering.py for targeted feature interventions.
    """

    def __init__(
        self,
        model: HookedTransformer,
        sae: AbstractSAE,
        hook_point: str,
        device: str = "cpu",
        preserve_error: bool = False,
        feature_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        resolved_device = resolve_device(device)
        self.model = model.to(resolved_device)
        self.sae = sae.to(resolved_device)
        self.hook_point = hook_point
        self.device = resolved_device
        self.preserve_error = preserve_error
        self.feature_transform = feature_transform
        self.latest_features: Optional[torch.Tensor] = None
        self.latest_reconstruction: Optional[torch.Tensor] = None

    @property
    def sae_dtype(self) -> torch.dtype:
        return next(self.sae.parameters()).dtype

    def check_dimensions(self) -> None:
        """Run a light sanity check against the model config."""
        if self.sae.d_in != self.model.cfg.d_model:
            raise ValueError(
                f"SAE d_in={self.sae.d_in} but model d_model={self.model.cfg.d_model}. "
                "For non-residual hooks, pass the hook's activation width as d_in."
            )

    def collect_activations(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return activations at this connector's hook point without modifying the model."""
        _, cache = self.model.run_with_cache(tokens.to(self.device), names_filter=self.hook_point)
        return cache[self.hook_point].detach().to(self.sae_dtype)

    def features_for_prompt(
        self,
        prompt: str,
        prepend_bos: bool = True,
        sae_projector: Optional[FeatureProjector] = None,
        projector_token_index: int = -1,
        projector_location: ProjectorLocation = "pre_activation",
    ) -> torch.Tensor:
        """
        Return SAE feature activations for each token in a prompt.

        If sae_projector is provided, it is applied to one token's feature
        vector either before or after the SAE activation function sparsifies it.

        The returned tensor has shape [batch, position, d_sae]. For a single
        prompt, batch is 1.
        """
        tokens = self.model.to_tokens(prompt, prepend_bos=prepend_bos).to(self.device)
        was_training = self.sae.training
        self.sae.eval()
        with torch.no_grad():
            activations = self.collect_activations(tokens)
            features = self._encode_features(
                activations,
                sae_projector,
                projector_token_index,
                projector_location,
            )
        if was_training:
            self.sae.train()
        return features.detach()

    def _apply_projector_to_token(
        self,
        vectors: torch.Tensor,
        sae_projector: FeatureProjector,
        projector_token_index: int,
    ) -> torch.Tensor:
        projected = vectors.clone()
        token_vectors = vectors[:, projector_token_index, :]
        projected_vectors = torch.stack([sae_projector(vector) for vector in token_vectors])
        if projected_vectors.shape != token_vectors.shape:
            raise ValueError(
                "sae_projector must return a vector with the same shape as its input: "
                f"got {tuple(projected_vectors.shape)}, expected {tuple(token_vectors.shape)}."
            )
        projected[:, projector_token_index, :] = projected_vectors
        return projected

    def _encode_features(
        self,
        acts: torch.Tensor,
        sae_projector: Optional[FeatureProjector] = None,
        projector_token_index: int = -1,
        projector_location: ProjectorLocation = "pre_activation",
    ) -> torch.Tensor:
        if sae_projector is None:
            return self.sae.encode(acts)
        if projector_location == "post_activation":
            features = self.sae.encode(acts)
            return self._apply_projector_to_token(features, sae_projector, projector_token_index)
        if projector_location != "pre_activation":
            raise ValueError(
                "projector_location must be 'pre_activation' or 'post_activation', "
                f"got {projector_location!r}."
            )
        encode_with_projector = getattr(self.sae, "encode_with_projector", None)
        if encode_with_projector is None:
            raise TypeError(
                f"{type(self.sae).__name__} does not support pre-activation projection. "
                "Use TrainableSAE or implement encode_with_projector."
            )

        def token_projector(pre_activations: torch.Tensor) -> torch.Tensor:
            return self._apply_projector_to_token(
                pre_activations,
                sae_projector,
                projector_token_index,
            )

        return encode_with_projector(acts, token_projector)

    def hook(
        self,
        mode: str = "reconstruct",
        sae_projector: Optional[FeatureProjector] = None,
        projector_token_index: int = -1,
        projector_location: ProjectorLocation = "pre_activation",
    ) -> Callable:
        sae = self.sae

        def hook_fn(acts: torch.Tensor, hook) -> torch.Tensor:
            del hook
            input_dtype = acts.dtype
            sae_acts = acts.to(self.sae_dtype)
            original_features = sae.encode(sae_acts)
            original_recon = sae.decode(original_features)
            features = self._encode_features(
                sae_acts,
                sae_projector,
                projector_token_index,
                projector_location,
            )

            if self.feature_transform is not None:
                features = self.feature_transform(features)

            recon = sae.decode(features)
            self.latest_features = features.detach()
            self.latest_reconstruction = recon.detach()

            if mode == "cache":
                return acts
            if mode == "reconstruct" and not self.preserve_error:
                return recon.to(input_dtype)
            if mode == "reconstruct":
                sae_error = sae_acts - original_recon
                return (recon + sae_error).to(input_dtype)
            raise ValueError(f"Unknown connector mode {mode!r}: use reconstruct or cache")

        return hook_fn

    def run_with_sae(
        self,
        tokens: torch.Tensor,
        mode: str = "reconstruct",
        sae_projector: Optional[FeatureProjector] = None,
        projector_token_index: int = -1,
        projector_location: ProjectorLocation = "pre_activation",
        **forward_kwargs,
    ):
        """Run the model with this SAE temporarily inserted at hook_point."""
        try:
            self.model.add_hook(
                self.hook_point,
                self.hook(
                    mode=mode,
                    sae_projector=sae_projector,
                    projector_token_index=projector_token_index,
                    projector_location=projector_location,
                ),
            )
            return self.model(tokens.to(self.device), **forward_kwargs)
        finally:
            self.model.reset_hooks()

    def _clean_generated_text(self, text: str) -> str:
        return text.replace("<bos>", "").replace("<eos>", "").strip()

    def generate_with_sae(
        self,
        prompt: str,
        mode: str = "reconstruct",
        sae_projector: Optional[FeatureProjector] = None,
        projector_token_index: int = -1,
        projector_location: ProjectorLocation = "pre_activation",
        clean: bool = True,
        **generate_kwargs,
    ) -> str:
        """Generate text with the SAE temporarily inserted."""
        tokens = self.model.to_tokens(prompt).to(self.device)
        try:
            self.model.add_hook(
                self.hook_point,
                self.hook(
                    mode=mode,
                    sae_projector=sae_projector,
                    projector_token_index=projector_token_index,
                    projector_location=projector_location,
                ),
            )
            out = self.model.generate(tokens, **generate_kwargs)
        finally:
            self.model.reset_hooks()
        text = self.model.to_string(out[0, tokens.shape[1]:])
        return self._clean_generated_text(text) if clean else text

    def generate(
        self,
        prompt: str,
        sae_projector: Optional[FeatureProjector] = None,
        projector_token_index: int = -1,
        projector_location: ProjectorLocation = "pre_activation",
        clean: bool = True,
        **generate_kwargs,
    ) -> str:
        """Generate text while optionally projecting one SAE feature vector."""
        return self.generate_with_sae(
            prompt,
            mode="reconstruct",
            sae_projector=sae_projector,
            projector_token_index=projector_token_index,
            projector_location=projector_location,
            clean=clean,
            **generate_kwargs,
        )


def load_hooked_transformer(
    model_name: str = "gpt2",
    device: str = "auto",
    **kwargs,
) -> HookedTransformer:
    """Load a TransformerLens model using the same package style as this repo."""
    _patch_torch_metadata_version()
    from transformer_lens import HookedTransformer

    resolved_device = resolve_device(device)
    tokenizer = kwargs.pop("tokenizer", None)
    if tokenizer is None and kwargs.get("local_files_only", False):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        tokenizer_path = snapshot_download(
            repo_id=model_name,
            local_files_only=True,
            allow_patterns=[
                "tokenizer*",
                "*.model",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "added_tokens.json",
            ],
        )
        tokenizer_kwargs = {
            "add_bos_token": kwargs.get("default_prepend_bos", True),
            "local_files_only": True,
            "trust_remote_code": kwargs.get("trust_remote_code", False),
            "use_fast": "phi" not in model_name.lower(),
        }
        if "token" in kwargs:
            tokenizer_kwargs["token"] = kwargs["token"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)

    try:
        model = HookedTransformer.from_pretrained(model_name, tokenizer=tokenizer, **kwargs)
    except OSError as exc:
        error_text = _exception_chain_text(exc)
        if "gated repositories" in error_text or "403 Forbidden" in error_text:
            raise OSError(
                f"Could not load {model_name!r} from Hugging Face. This model appears "
                "to require gated-repo access. Accept the model terms on Hugging Face, "
                "then use a token with access to public gated repositories via "
                "`huggingface-cli login` or the `HF_TOKEN` environment variable."
            ) from exc
        raise
    model.to(resolved_device)
    model.eval()
    return model


def activation_batches_from_texts(
    connector: SAEConnector,
    texts: Iterable[str],
    batch_size_tokens: int = 4096,
) -> Iterator[torch.Tensor]:
    """
    Convert texts into flattened activation batches for SAE training.

    This is simple and good for small experiments.  For large runs, use SAELens'
    activation store/training runner or adapt this generator to stream a dataset.
    """
    buffer: List[torch.Tensor] = []
    buffered_tokens = 0

    for text in texts:
        tokens = connector.model.to_tokens(text).to(connector.device)
        acts = connector.collect_activations(tokens).reshape(-1, connector.sae.d_in)
        buffer.append(acts)
        buffered_tokens += acts.shape[0]

        if buffered_tokens >= batch_size_tokens:
            joined = torch.cat(buffer, dim=0)
            yield joined[:batch_size_tokens]
            remainder = joined[batch_size_tokens:]
            buffer = [remainder] if remainder.numel() else []
            buffered_tokens = remainder.shape[0] if remainder.numel() else 0

    if buffer:
        yield torch.cat(buffer, dim=0)


def fit_sae_on_texts(
    sae: TrainableSAE,
    connector: SAEConnector,
    texts: Iterable[str],
    steps: int = 100,
    batch_size_tokens: int = 4096,
    optimizer: Optional[torch.optim.Optimizer] = None,
    record_nonzero_features: bool = False,
    sparsity_udf: Optional[Callable[[TrainableSAE, Dict[str, float]], None]] = None,
) -> List[Dict[str, float]]:
    """Small training loop for quick custom SAE experiments."""
    optimizer = optimizer or torch.optim.AdamW(sae.parameters(), lr=sae.cfg.lr)
    metrics: List[Dict[str, float]] = []

    for step, batch in enumerate(
        activation_batches_from_texts(connector, texts, batch_size_tokens=batch_size_tokens),
        start=1,
    ):
        metrics.append(
            sae.training_step(
                batch.to(connector.device),
                optimizer,
                record_nonzero_features=record_nonzero_features,
                sparsity_udf=sparsity_udf,
            )
        )
        if step >= steps:
            break

    return metrics

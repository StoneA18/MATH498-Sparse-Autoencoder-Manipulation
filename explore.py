"""
explore.py — Interactive REPL for exploring LLM features and steering.

Run with:  python explore.py

Type 'help' inside the REPL for a full command list.
Type 'help <command>' for detailed usage of a specific command.
"""

import sys
import torch

from feature_steering import (
    load_model_and_sae,
    load_sae_from_name,
    format_prompt,
    top_features_for_text,
    token_feature_activations,
    generate_exploration_html,
    FeatureSteerer,
    train_custom_sae,
    is_tbk_sae,
    MODELS,
    DEFAULT_MODEL,
)


def p(msg=""):
    print(msg)


def npedia(feature_id: int, state: dict) -> str:
    base = f"https://neuronpedia.org/{state['config'].neuronpedia_id}"
    return f"{base}/{feature_id}"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_text(arg: str, usage_hint: str):
    arg = arg.strip()

    if arg.startswith("-f "):
        filepath = arg[3:].strip()
        if not filepath:
            print(f"Error: {usage_hint}")
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                text = fh.read()
        except FileNotFoundError:
            print(f"Error: file not found: {filepath!r}")
            return None
        except OSError as e:
            print(f"Error: could not read file: {e}")
            return None
        label = f"(from file: {filepath})"
    else:
        text = arg
        label = ""

    if not text.strip():
        print(f"Error: {usage_hint}")
        return None

    return text, label


def extract_flag(args: str, flag: str):
    tokens = args.split()
    if flag not in tokens:
        return args, None

    idx = tokens.index(flag)
    if idx + 1 >= len(tokens):
        print(f"Error: {flag} requires a value argument")
        return args, None

    value = tokens[idx + 1]
    remaining_tokens = tokens[:idx] + tokens[idx + 2:]
    return " ".join(remaining_tokens), value


def _fmt_entry(entry: tuple) -> str:
    """Format a clamp entry tuple as a human-readable string."""
    op = entry[0]
    if op == 'cond_dist':
        _, prob, dist, *params = entry
        param_str = " ".join(str(p) for p in params)
        return f"cond_dist  p={prob}  {dist}({param_str})"
    return f"{op:<8} {entry[1]}"


def _restore_clamp(steerer, feat_id: int, entry: tuple):
    """Re-apply a clamp entry to the steerer (used in compare after clearing)."""
    op = entry[0]
    if op == 'clamp':
        steerer.clamp(feat_id, entry[1])
    elif op == 'add':
        steerer.add(feat_id, entry[1])
    elif op == 'scale':
        steerer.scale(feat_id, entry[1])
    elif op == 'cond_dist':
        _, prob, dist, *params = entry
        steerer.cond_dist(feat_id, prob, dist, *params)


def write_output(filepath: str, content: str):
    try:
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(content)
        print(f"Output written to {filepath!r}")
    except OSError as e:
        print(f"Error: could not write to {filepath!r}: {e}")


def _parse_n_flag(args: str, state: dict):
    args, n_str = extract_flag(args, "-n")
    if n_str is not None:
        try:
            return args, int(n_str)
        except ValueError:
            print(f"Error: -n requires an integer, got {n_str!r} — using model default")
    return args, state["config"].default_max_tokens


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_analyze(args: str, state: dict):
    tbk = "-tbk" in args.split() or is_tbk_sae(state["sae"])
    if "-tbk" in args.split():
        args = " ".join(t for t in args.split() if t != "-tbk")

    result = load_text(args, "Usage: analyze <text> [-tbk]\n       analyze -f <filepath> [-tbk]")
    if result is None:
        return
    text, label = result

    preview = text[:80].replace("\n", " ") + ("..." if len(text) > 80 else "")
    if label:
        print(label)
    print(f"Analyzing: {preview!r}" + ("  [tbk]" if tbk else ""))

    results = top_features_for_text(
        text, state["model"], state["sae"], state["config"].hook_point,
        k=10, device=state["device"], selection="tbk" if tbk else "topk",
    )

    print(f"\nTop features for: {preview!r}")
    print(f"  {'Rank':<6} {'Feature ID':<12} {'Activation':<12} Neuronpedia")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*40}")
    for rank, (feat_id, val) in enumerate(results, 1):
        print(f"  {rank:<6} {feat_id:<12} {val:<12.3f} {npedia(feat_id, state)}")
    print()


def cmd_feature(args: str, state: dict):
    parts = args.strip().split(None, 1)
    if len(parts) < 2:
        print("Usage: feature <id> <text>")
        print("       feature <id> -f <filepath>")
        return

    try:
        feat_id = int(parts[0])
    except ValueError:
        print(f"Error: feature ID must be an integer, got: {parts[0]!r}")
        return

    result = load_text(parts[1], "Usage: feature <id> <text>\n       feature <id> -f <filepath>")
    if result is None:
        return
    text, label = result

    if label:
        print(label)

    results = token_feature_activations(
        text, feat_id, state["model"], state["sae"],
        state["config"].hook_point, state["device"],
    )

    max_val = max((v for _, v in results), default=1.0)

    print(f"\nFeature {feat_id} — per-token activation")
    print(f"  {'Pos':<5} {'Token':<18} {'Activation':<12} Intensity")
    print(f"  {'-'*5} {'-'*18} {'-'*12} {'-'*20}")
    for i, (token, val) in enumerate(results):
        bar_len = int((val / max(max_val, 1e-6)) * 20) if val > 0 else 0
        bar = "#" * bar_len if val > 0 else "."
        print(f"  {i:<5} {repr(token):<18} {val:<12.4f} {bar}")

    print(f"\nNeuronpedia: {npedia(feat_id, state)}")
    print()


def cmd_clamp(args: str, state: dict):
    parts = args.strip().split()
    if not parts:
        print("Usage: clamp <feature_id> [value]")
        return

    try:
        feat_id = int(parts[0])
    except ValueError:
        print(f"Error: feature ID must be an integer, got: {parts[0]!r}")
        return

    value = 20.0
    if len(parts) > 1:
        try:
            value = float(parts[1])
        except ValueError:
            print(f"Error: value must be a number, got: {parts[1]!r}")
            return

    state["steerer"].clamp(feat_id, value)
    print(f"Clamped feature {feat_id} = {value}")
    print(f"  {npedia(feat_id, state)}")
    print()


def cmd_add(args: str, state: dict):
    parts = args.strip().split()
    if not parts:
        print("Usage: add <feature_id> [value]")
        return

    try:
        feat_id = int(parts[0])
    except ValueError:
        print(f"Error: feature ID must be an integer, got: {parts[0]!r}")
        return

    value = 10.0
    if len(parts) > 1:
        try:
            value = float(parts[1])
        except ValueError:
            print(f"Error: value must be a number, got: {parts[1]!r}")
            return

    state["steerer"].add(feat_id, value)
    print(f"Adding {value} to feature {feat_id}")
    print(f"  {npedia(feat_id, state)}")
    print()


def cmd_scale(args: str, state: dict):
    parts = args.strip().split()
    if not parts:
        print("Usage: scale <feature_id> [factor]")
        return

    try:
        feat_id = int(parts[0])
    except ValueError:
        print(f"Error: feature ID must be an integer, got: {parts[0]!r}")
        return

    factor = 2.0
    if len(parts) > 1:
        try:
            factor = float(parts[1])
        except ValueError:
            print(f"Error: factor must be a number, got: {parts[1]!r}")
            return

    state["steerer"].scale(feat_id, factor)
    print(f"Scaling feature {feat_id} by {factor}")
    print(f"  {npedia(feat_id, state)}")
    print()


def cmd_cond_dist(args: str, state: dict):
    # Expected: <feature_id> <prob> <dist> <param1> [<param2> ...]
    # e.g.      18493 0.5 normal 40 10
    parts = args.strip().split()
    usage = (
        "Usage: cond_dist <id> <prob> normal <mean> <std>\n"
        "       cond_dist <id> <prob> uniform <low> <high>"
    )

    if len(parts) < 4:
        print(usage)
        return

    try:
        feat_id = int(parts[0])
    except ValueError:
        print(f"Error: feature ID must be an integer, got: {parts[0]!r}")
        return

    try:
        prob = float(parts[1])
        if not (0.0 <= prob <= 1.0):
            raise ValueError
    except ValueError:
        print(f"Error: prob must be a float in [0, 1], got: {parts[1]!r}")
        return

    dist = parts[2].lower()
    param_strs = parts[3:]

    try:
        dist_params = tuple(float(p) for p in param_strs)
    except ValueError:
        print(f"Error: distribution parameters must be numbers")
        return

    try:
        state["steerer"].cond_dist(feat_id, prob, dist, *dist_params)
    except ValueError as e:
        print(f"Error: {e}")
        return

    param_desc = " ".join(str(p) for p in dist_params)
    print(f"cond_dist feature {feat_id}: p={prob}  {dist}({param_desc})")
    print(f"  {npedia(feat_id, state)}")
    print()


def cmd_unclamp(args: str, state: dict):
    parts = args.strip().split()
    if not parts:
        print("Usage: unclamp <feature_id>")
        return

    try:
        feat_id = int(parts[0])
    except ValueError:
        print(f"Error: feature ID must be an integer, got: {parts[0]!r}")
        return

    steerer = state["steerer"]
    if feat_id in steerer.clamped_features:
        steerer.unclamp(feat_id)
        print(f"Feature {feat_id} released.")
    else:
        print(f"Feature {feat_id} was not clamped.")
    print()


def cmd_list(state: dict):
    clamps = state["steerer"].list_clamps()
    if not clamps:
        print("No features currently clamped.")
        print()
        return

    print("Active feature clamps:")
    print(f"  {'Feature ID':<12} {'Setting':<30} Neuronpedia")
    print(f"  {'-'*12} {'-'*30} {'-'*40}")
    for feat_id, entry in sorted(clamps.items()):
        print(f"  {feat_id:<12} {_fmt_entry(entry):<30} {npedia(feat_id, state)}")
    print()


def cmd_generate(args: str, state: dict):
    args, outpath = extract_flag(args, "-o")
    args, max_new_tokens = _parse_n_flag(args, state)
    result = load_text(
        args,
        "Usage: generate <prompt> [-n <tokens>] [-o <outfile>]\n"
        "       generate -f <filepath> [-n <tokens>] [-o <outfile>]",
    )
    if result is None:
        return
    prompt, src_label = result

    formatted_prompt = format_prompt(prompt, state["config"])

    steerer = state["steerer"]
    clamps = steerer.list_clamps()
    status = f"Generating with {len(clamps)} clamped feature(s)..." if clamps else "Generating (no features clamped)..."
    if src_label:
        print(src_label)
    print(f"{status} (max {max_new_tokens} tokens)")

    generated = steerer.generate(formatted_prompt, max_new_tokens=max_new_tokens, verbose=False)

    print(f"\nPrompt:    {prompt}")
    print(f"Generated: {generated}")
    print()

    if outpath:
        clamp_summary = ", ".join(f"{fid} {_fmt_entry(e)}" for fid, e in sorted(clamps.items())) or "none"
        write_output(outpath, (
            f"=== generate output ===\n"
            f"model: {state['config'].display_name}\n"
            f"max_new_tokens: {max_new_tokens}\n"
            f"clamped features: {clamp_summary}\n\n"
            f"--- prompt ---\n{prompt}\n\n"
            f"--- generated ---\n{generated}\n"
        ))


def cmd_compare(args: str, state: dict):
    args, outpath = extract_flag(args, "-o")
    args, max_new_tokens = _parse_n_flag(args, state)
    result = load_text(
        args,
        "Usage: compare <prompt> [-n <tokens>] [-o <outfile>]\n"
        "       compare -f <filepath> [-n <tokens>] [-o <outfile>]",
    )
    if result is None:
        return
    prompt, src_label = result

    if src_label:
        print(src_label)

    formatted_prompt = format_prompt(prompt, state["config"])

    steerer = state["steerer"]
    clamps = steerer.list_clamps()
    if not clamps:
        print("Warning: no features clamped — both outputs will be identical.")
    print(f"(max {max_new_tokens} tokens per generation)")

    print("Generating normal output...")
    steerer.unclamp_all()
    normal_out = steerer.generate(formatted_prompt, max_new_tokens=max_new_tokens, verbose=False)

    for feat_id, entry in clamps.items():
        _restore_clamp(steerer, feat_id, entry)

    print("Generating steered output...")
    try:
        steered_out = steerer.generate(formatted_prompt, max_new_tokens=max_new_tokens, verbose=False)
        steered_err = None
    except Exception as e:
        steered_out = ""
        steered_err = str(e)
        print(f"Error: steered generation failed: {e}")

    print(f"\nPrompt: {prompt}")
    print(f"\n--- normal (no clamps) ---")
    print(normal_out)
    print(f"\n--- steered ({len(clamps)} clamp(s)) ---")
    print(steered_out if steered_out else f"(error: {steered_err})")
    print()

    if outpath:
        clamp_summary = ", ".join(f"{fid} {_fmt_entry(e)}" for fid, e in sorted(clamps.items())) or "none"
        steered_section = steered_out if steered_out else (f"[error: {steered_err}]" if steered_err else "(empty)")
        write_output(outpath, (
            f"=== compare output ===\n"
            f"model: {state['config'].display_name}\n"
            f"max_new_tokens: {max_new_tokens}\n"
            f"clamped features: {clamp_summary}\n\n"
            f"--- prompt ---\n{prompt}\n\n"
            f"--- normal (no clamps) ---\n{normal_out}\n\n"
            f"--- steered ({len(clamps)} clamp(s)) ---\n{steered_section}\n"
        ))


def cmd_explore(args: str, state: dict):
    """
    explore <text>
    explore -f <filepath> [-topn <n>] [-o <outfile>]

    Generate a self-contained HTML report showing the top-N activated features
    for each token.  Click any feature cell to see its full activation profile
    across all tokens.  Only features that appear in the top-N for at least one
    token are included.
    """
    usage = (
        "Usage: explore <text> [-topn <n>] [-o <outfile>]\n"
        "       explore -f <filepath> [-topn <n>] [-o <outfile>]"
    )

    args, outpath = extract_flag(args, "-o")
    args, topn_str = extract_flag(args, "-topn")

    tbk = "-tbk" in args.split() or is_tbk_sae(state["sae"])
    if "-tbk" in args.split():
        args = " ".join(t for t in args.split() if t != "-tbk")

    try:
        top_n = int(topn_str) if topn_str else 10
        if top_n < 1:
            print("Error: -topn must be >= 1")
            return
    except ValueError:
        print(f"Error: -topn must be an integer, got {topn_str!r}")
        return

    result = load_text(args, usage)
    if result is None:
        return
    text, src_label = result

    if src_label:
        print(src_label)

    outpath = outpath or "explorationoutput.html"

    preview = text[:60].replace("\n", " ") + ("..." if len(text) > 60 else "")
    print(f"Running feature exploration on: {preview!r}")
    print(f"  top_n={top_n}  selection={'tbk' if tbk else 'topk'}  output={outpath!r}")

    html = generate_exploration_html(
        text=text,
        model=state["model"],
        sae=state["sae"],
        hook_point=state["config"].hook_point,
        top_n=top_n,
        device=state["device"],
        model_name=state["config"].display_name,
        selection="tbk" if tbk else "topk",
    )

    write_output(outpath, html)
    n_tokens = len(state["model"].to_str_tokens(text))
    print(f"Report written: {n_tokens} tokens, top {top_n} features each.")
    print()


def cmd_load_sae(args: str, state: dict):
    name = args.strip()
    if not name:
        print("Usage: load_sae <name>")
        print("  <name> is the identifier used with train_sae -o")
        print()
        return

    print(f"Loading SAE {name!r}...")
    try:
        sae, hook_point = load_sae_from_name(name, device=state["device"])
    except Exception as e:
        print(f"Error: {e}")
        print()
        return

    import dataclasses
    state["sae"]     = sae
    state["config"]  = dataclasses.replace(
        state["config"],
        hook_point=hook_point,
        neuronpedia_id=None,   # custom SAE — neuronpedia links won't apply
    )
    state["steerer"] = FeatureSteerer(
        state["model"], sae, hook_point, device=state["device"]
    )

    print("Custom SAE active. analyze / feature / clamp / generate all use it now.")
    print("Note: neuronpedia links will be blank — this SAE has no online listing.")
    print()


def cmd_train_sae(args: str, state: dict):
    """
    Parse and dispatch: train_sae <layer> [options...]

    Flags
    -----
      -hook  resid_post|resid_pre|mlp_out|attn_out   (default: resid_post)
      -d     <d_sae>          dictionary size; overrides -e when set
      -e     <expansion>      d_sae = d_in × expansion  (default: 16)
      -act   relu|topk|tbk    activation function        (default: topk)
      -k     <int>            active features for topk   (default: 50)
      -l1    <float>          L1 coefficient for relu    (default: 5e-5)
      -lr    <float>          learning rate              (default: 2e-4)
      -tokens <int>           total training tokens      (default: 500000)
      -bs    <int>            batch size in tokens       (default: 4096)
      -ctx   <int>            context / sequence length  (default: 128)
      -ds    <str>            HuggingFace dataset path   (default: NeelNanda/pile-10k)
      -o     <name>           name for this SAE          (default: sae_<layer>)
      -wandb                  enable Weights & Biases logging
    """
    usage = (
        "Usage: train_sae <layer> [options]\n"
        "  -hook resid_post|resid_pre|mlp_out|attn_out\n"
        "  -d <d_sae>   -e <expansion_factor>\n"
        "  -act relu|topk   -k <k>   -l1 <coeff>\n"
        "  -lr <rate>   -tokens <n>   -bs <batch_tokens>   -ctx <seq_len>\n"
        "  -ds <dataset_path>   -o <name>   -wandb"
    )

    parts = args.strip().split()
    if not parts:
        print(usage)
        return

    try:
        layer = int(parts[0])
    except ValueError:
        print(f"Error: layer must be an integer, got {parts[0]!r}")
        return

    # ---- parse remaining flags -----------------------------------------------
    # Build a mutable token list (skip parts[0] already consumed)
    tokens = parts[1:]

    def _pop_flag(name: str):
        """Return value after flag and remove both from tokens, or None."""
        if name in tokens:
            idx = tokens.index(name)
            if idx + 1 >= len(tokens):
                print(f"Error: {name} requires a value")
                return None, True   # (value, error)
            val = tokens[idx + 1]
            del tokens[idx:idx + 2]
            return val, False
        return None, False

    def _pop_bool_flag(name: str) -> bool:
        if name in tokens:
            tokens.remove(name)
            return True
        return False

    hook_type_str,   err = _pop_flag("-hook");
    if err: return
    d_sae_str,       err = _pop_flag("-d");
    if err: return
    exp_str,         err = _pop_flag("-e");
    if err: return
    act_str,         err = _pop_flag("-act");
    if err: return
    k_str,           err = _pop_flag("-k");
    if err: return
    l1_str,          err = _pop_flag("-l1");
    if err: return
    lr_str,          err = _pop_flag("-lr");
    if err: return
    tokens_str,      err = _pop_flag("-tokens");
    if err: return
    bs_str,          err = _pop_flag("-bs");
    if err: return
    ctx_str,         err = _pop_flag("-ctx");
    if err: return
    ds_str,          err = _pop_flag("-ds");
    if err: return
    out_str,         err = _pop_flag("-o");
    if err: return
    log_wandb             = _pop_bool_flag("-wandb")

    if tokens:
        print(f"Error: unrecognised arguments: {' '.join(tokens)}")
        print(usage)
        return

    # ---- type-check and apply defaults ---------------------------------------
    hook_type        = hook_type_str or "resid_post"
    activation_fn    = act_str or "topk"
    dataset_path     = ds_str or "NeelNanda/pile-10k"
    sae_name         = out_str or f"sae_{layer}"

    try:
        expansion_factor = int(exp_str) if exp_str else 16
        d_sae            = int(d_sae_str) if d_sae_str else None
        k                = int(k_str) if k_str else 50
        l1_coefficient   = float(l1_str) if l1_str else 5e-5
        lr               = float(lr_str) if lr_str else 2e-4
        training_tokens  = int(tokens_str) if tokens_str else 500_000
        batch_size       = int(bs_str) if bs_str else 4096
        context_size     = int(ctx_str) if ctx_str else 128
    except ValueError as e:
        print(f"Error: bad numeric argument: {e}")
        return

    valid_hooks = ("resid_post", "resid_pre", "mlp_out", "attn_out")
    if hook_type not in valid_hooks:
        print(f"Error: -hook must be one of {valid_hooks}, got {hook_type!r}")
        return

    if activation_fn not in ("relu", "topk", "tbk"):
        print(f"Error: -act must be 'relu', 'topk', or 'tbk', got {activation_fn!r}")
        return

    if layer < 0:
        print(f"Error: layer must be >= 0, got {layer}")
        return

    n_layers = state["model"].cfg.n_layers
    if layer >= n_layers:
        print(f"Error: layer {layer} out of range for this model (0–{n_layers - 1})")
        return

    # ---- run training --------------------------------------------------------
    try:
        trained_sae = train_custom_sae(
            model=state["model"],
            config_name=state["config_name"],
            layer=layer,
            name=sae_name,
            hook_type=hook_type,
            expansion_factor=expansion_factor,
            d_sae=d_sae,
            activation_fn=activation_fn,
            k=k,
            l1_coefficient=l1_coefficient,
            lr=lr,
            training_tokens=training_tokens,
            train_batch_size_tokens=batch_size,
            context_size=context_size,
            dataset_path=dataset_path,
            device=state["device"],
            log_to_wandb=log_wandb,
        )
    except (ImportError, ValueError) as e:
        print(f"Error: {e}")
        return

    print(f"Training finished. SAE has {trained_sae.W_enc.shape[1]:,} features.")
    print(f"Load it with:  load_sae {sae_name}")
    print()


def cmd_models(_args, _state):
    print("Available models:")
    print(f"  {'Key':<22} {'Chat':<6} {'Def.tok':<9} Description")
    print(f"  {'-'*22} {'-'*6} {'-'*9} {'-'*40}")
    for key, cfg in MODELS.items():
        chat = "yes" if cfg.is_chat else "no"
        print(f"  {key:<22} {chat:<6} {cfg.default_max_tokens:<9} {cfg.display_name}")
    print("\nSwitch with: model <key>")
    print()


def cmd_model(args: str, state: dict):
    key = args.strip()
    if not key:
        print(f"Current model: {state['config_name']}  (use 'models' to list options)")
        print()
        return

    if key == "-layer":
        n = state["model"].cfg.n_layers
        print(f"{state['config_name']} has {n} layers  (valid SAE layers: 0 – {n - 1})")
        print()
        return

    if key not in MODELS:
        print(f"Error: unknown model {key!r}. Available: {', '.join(MODELS)}")
        print()
        return

    if key == state["config_name"]:
        print(f"Already using {key!r}.")
        print()
        return

    print(f"Switching to {MODELS[key].display_name}...")
    print("(This will download weights on first use.)")

    import gc
    model, sae, config = load_model_and_sae(key, device=state["device"])

    for k in ("model", "sae", "steerer"):
        state.pop(k, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    state["model"]       = model
    state["sae"]         = sae
    state["config"]      = config
    state["config_name"] = key
    state["steerer"]     = FeatureSteerer(model, sae, config.hook_point, device=state["device"])

    print(f"Switched to {config.display_name}")
    print()


def cmd_tokens(args: str, state: dict):
    parts = args.strip().split()
    if not parts:
        print(f"Current default max tokens: {state['config'].default_max_tokens}")
        print()
        return

    try:
        new_val = int(parts[0])
        if new_val <= 0:
            print("Error: max tokens must be positive")
            return
        state["config"].default_max_tokens = new_val
        print(f"Default max tokens set to {new_val}")
        print()
    except ValueError:
        print(f"Error: must be an integer, got {parts[0]!r}")
        print()


_DETAILED_HELP = {
    "explore": """\
explore <text> [-topn <n>] [-tbk] [-o <file>]
explore -f <path> [-topn <n>] [-tbk] [-o <file>]

  Generate a self-contained HTML report showing the top-N activated features
  for each token in the input text.

  Open the file in a browser:
    - Each row is a token; columns are Rank 1 .. N.
    - Cells are colour-coded by activation strength.
    - Click any cell to open a detail panel showing that feature's activation
      at every token position (with a bar chart).
    - Use the search box to jump to any feature by ID.
    - Only features that appear in the top-N for at least one token are included.

  Flags:
    -topn <n>   number of top features per token  (default: 10)
    -tbk        top-bottom-k: select by greatest magnitude instead of highest
                value — useful when activations can be negative
    -o <file>   output HTML file                  (default: explorationoutput.html)
    -f <path>   read input text from a file
""",

    "analyze": """\
analyze <text> [-tbk]
analyze -f <path> [-tbk]

  Find the 10 features most strongly activated by the input text and print
  them with their activation values and Neuronpedia links.

  Aggregation is by peak activation across token positions (max pooling).

  Flags:
    -tbk   top-bottom-k: rank by greatest absolute magnitude instead of highest
           value — returns signed activations, so negatives may appear
""",

    "feature": """\
feature <id> <text>
feature <id> -f <path>

  Show per-token activation of a single feature across the input text.
  Useful for verifying which words or positions trigger a feature.
  Prints a bar next to each token proportional to its activation value.
""",

    "clamp": """\
clamp <id> [value]

  Force feature <id> to exactly <value> at every token during generation.
  Default value: 20.0.  Typical active range is 1–30; >100 can cause incoherence.
  Use 'generate' or 'compare' to see the effect.
""",

    "add": """\
add <id> [value]

  Add <value> to the natural activation of feature <id> at every token.
  Default value: 10.0.  Unlike clamp, this preserves the relative variation.
""",

    "scale": """\
scale <id> [factor]

  Multiply the activation of feature <id> by <factor> at every token.
  Default factor: 2.0.
""",

    "cond_dist": """\
cond_dist <id> <prob> normal <mean> <std>
cond_dist <id> <prob> uniform <low> <high>

  Per-token stochastic intervention: with probability <prob>, replace the
  feature's activation with a sample from the given distribution.

  Distributions:
    normal  <mean> <std>    e.g. cond_dist 18493 0.5 normal 40 10
    uniform <low>  <high>   e.g. cond_dist 18493 0.3 uniform 5 30
""",

    "generate": """\
generate <prompt> [-n <tokens>] [-o <file>]
generate -f <path> [-n <tokens>] [-o <file>]

  Generate text from <prompt> with all active feature clamps applied.
  Flags:
    -n <tokens>  max new tokens to generate  (default: model default)
    -o <file>    write prompt + output to a file
    -f <path>    read the prompt from a file
""",

    "compare": """\
compare <prompt> [-n <tokens>] [-o <file>]
compare -f <path> [-n <tokens>] [-o <file>]

  Generate the same prompt twice — once with no clamps, once with all active
  clamps — and print both outputs side by side.

  Flags:
    -n <tokens>  max new tokens per generation  (default: model default)
    -o <file>    write both outputs to a file
    -f <path>    read the prompt from a file
""",

    "train_sae": """\
train_sae <layer> [-o <name>] [flags...]

  Train a custom Sparse Autoencoder on the currently loaded model.
  The finished SAE is saved to  custom_saes/<name>/  and can be loaded with
  'load_sae <name>'.

  Required:
    <layer>          transformer layer to attach the SAE to (0-indexed)

  Flags:
    -o <name>        name for this SAE               (default: sae_<layer>)
    -hook <type>     resid_post | resid_pre | mlp_out | attn_out
                                                     (default: resid_post)
    -e <int>         expansion factor: d_sae = d_in x e  (default: 16)
    -d <int>         dictionary size directly (overrides -e)
    -act topk|relu|tbk activation function           (default: topk)
                     tbk = top-bottom-k: selects k features by greatest
                     absolute magnitude; encoder values may be negative
    -k <int>         active features per token, topk only  (default: 50)
    -l1 <float>      L1 sparsity coefficient, relu only   (default: 5e-5)
    -lr <float>      learning rate                   (default: 2e-4)
    -tokens <int>    total training tokens            (default: 500000)
    -bs <int>        tokens per gradient step         (default: 4096)
    -ctx <int>       context length                   (default: 128)
    -ds <str>        HuggingFace dataset path         (default: NeelNanda/pile-10k)
    -wandb           enable Weights & Biases logging
""",

    "load_sae": """\
load_sae <name>

  Load a locally-trained SAE by name and make it active for all commands
  (analyze, feature, explore, clamp, generate, compare).

  <name> must be a directory under  custom_saes/  created by 'train_sae'.
  Run 'load_sae' with no argument to see which SAEs are available (will error
  with a list of known names).
""",

    "model": """\
model [key]
model -layer

  With no argument: print the currently loaded model.
  With a key: switch to that model (downloads weights on first use).
  -layer: print the number of layers in the current model and the valid
          layer range for SAE placement (0 to n_layers - 1).
  Use 'models' to list all available model keys.
""",

    "models": """\
models

  List all available model configurations with their key, chat flag,
  default token budget, and display name.  Switch with 'model <key>'.
""",

    "tokens": """\
tokens [number]

  With no argument: show the current default max-tokens budget for generate/compare.
  With a number: set it.  Can also be overridden per-command with -n.
""",
}


def print_help(args: str = ""):
    cmd = args.strip().lower()
    if cmd:
        if cmd in _DETAILED_HELP:
            print()
            print(_DETAILED_HELP[cmd])
        else:
            print(f"No detailed help for {cmd!r}.")
            print("Type 'help' (no argument) for the full command list.")
            print()
        return

    print("""
Commands                         (type 'help <command>' for details)
--------
  Analysis
    explore  <text|-f path>      HTML report: top-N features per token, interactive
    analyze  <text|-f path>      top 10 features for the whole text
    feature  <id> <text|-f path> per-token activation of one feature
    (add -tbk to explore/analyze to select by greatest magnitude instead of highest value)

  Steering
    clamp    <id> [value]        force a feature to a fixed value
    add      <id> [value]        add to a feature's activation
    scale    <id> [factor]       multiply a feature's activation
    cond_dist <id> <prob> ...    per-token stochastic intervention
    unclamp  <id>                remove clamp on one feature
    clear                        remove all clamps
    list                         show active clamps

  Generation
    generate <prompt|-f path>    generate with active clamps
    compare  <prompt|-f path>    normal vs. steered output side by side

  SAE management
    train_sae <layer>            train a custom SAE (many options)
    load_sae  <name>             load a trained SAE as the active SAE

  Model
    models                       list available models
    model    [key]               show or switch current model
    tokens   [n]                 show or set default generation budget

  Other
    rpt                          repeat last command
    help     [command]           this message, or detail for one command
    quit                         exit

Workflow:
  1. analyze <text>    find feature IDs that fire for your topic
  2. feature <id>      verify where a feature fires token-by-token
  3. explore -f <txt>  browse all top features interactively in a browser
  4. clamp <id> 25     force a feature on
  5. compare <prompt>  see normal vs steered output side by side
""")


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Welcome to SAE Manipulation Tool.")
    print("Type 'help' for commands, 'quit' to exit.")
    print()

    model, sae, config = load_model_and_sae(DEFAULT_MODEL, device=device)

    state = {
        "model":       model,
        "sae":         sae,
        "config":      config,
        "config_name": DEFAULT_MODEL,
        "steerer":     FeatureSteerer(model, sae, config.hook_point, device=device),
        "device":      device,
    }

    print("Ready. Type 'help' to see commands.")
    print()

    COMMANDS = {
        "analyze":  lambda args: cmd_analyze(args, state),
        "feature":  lambda args: cmd_feature(args, state),
        "clamp":    lambda args: cmd_clamp(args, state),
        "add":      lambda args: cmd_add(args, state),
        "scale":     lambda args: cmd_scale(args, state),
        "cond_dist": lambda args: cmd_cond_dist(args, state),
        "unclamp":   lambda args: cmd_unclamp(args, state),
        "clear":    lambda _:    (state["steerer"].unclamp_all(), print("All clamps removed.\n")),
        "list":     lambda _:    cmd_list(state),
        "generate": lambda args: cmd_generate(args, state),
        "compare":  lambda args: cmd_compare(args, state),
        "tokens":    lambda args: cmd_tokens(args, state),
        "models":    lambda args: cmd_models(args, state),
        "model":     lambda args: cmd_model(args, state),
        "explore":   lambda args: cmd_explore(args, state),
        "load_sae":  lambda args: cmd_load_sae(args, state),
        "train_sae": lambda args: cmd_train_sae(args, state),
        "help":     lambda args: print_help(args),
    }

    last_cmd, last_args = None, ""

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd  = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif cmd in ("rpt", "repeat"):
            if last_cmd is None:
                print("Nothing to repeat yet.")
                print()
                continue
            print(f"Repeating: {last_cmd} {last_args}")
            cmd, args = last_cmd, last_args

        if cmd in COMMANDS:
            try:
                COMMANDS[cmd](args)
            except Exception as e:
                print(f"Error: {e}")
                print()
            last_cmd, last_args = cmd, args
        elif cmd not in ("rpt", "repeat"):
            print(f"Unknown command: {cmd!r}  (type 'help' for a list)")
            print()


if __name__ == "__main__":
    main()

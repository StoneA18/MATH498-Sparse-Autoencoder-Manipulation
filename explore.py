"""
explore.py — Interactive REPL for exploring GPT-2 features and steering.

Run with:  python explore.py

Commands
--------
  analyze <text>          Show top 10 features activated by <text>
  analyze -f <path>       Same, but read the text from a file
  feature <id> <text>     Show per-token activation of feature <id> for <text>
  feature <id> -f <path>  Same, but read the text from a file
  clamp <id> [value]      Force feature <id> to <value> during generation (default: 20)
  unclamp <id>            Remove clamp on feature <id>
  clear                   Remove all clamps
  list                    Show currently clamped features
  generate <prompt>            Generate text with current clamps active
  generate -f <path>           Same, but read the prompt from a file
  generate ... -o <path>       Write generated output to a file
  compare <prompt>             Generate twice — once normal, once with clamps — side by side
  compare -f <path>            Same, but read the prompt from a file
  compare ... -o <path>        Write both outputs to a file
  rpt                     Repeat the last command
  help                    Show this message
  quit                    Exit

Suggested workflow
------------------
  1. analyze <some topic>      — find features that activate for your topic
  2. feature <id> <text>       — verify a feature fires where you expect
  3. Look it up: https://neuronpedia.org/<neuronpedia_id>/<id>
  4. clamp <id> 25             — force that feature on
  5. generate <prompt>         — see the effect on generation
  6. compare <prompt>          — see normal vs. steered side by side
  7. clear                     — reset and try something different
"""

import sys
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

from feature_steering import (
    load_model_and_sae,
    format_prompt,
    top_features_for_text,
    token_feature_activations,
    FeatureSteerer,
    MODELS,
    DEFAULT_MODEL,
)

console = Console()


def npedia(feature_id: int, state: dict) -> str:
    """Format a Neuronpedia URL for a given feature using the active model's ID."""
    base = f"https://neuronpedia.org/{state['config'].neuronpedia_id}"
    return f"{base}/{feature_id}"


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def load_text(arg: str, usage_hint: str) -> tuple[str, str] | None:
    """
    Parse a text-or-file argument and return (text, source_label).

    If arg starts with "-f " the remainder is treated as a filepath and the
    file's contents are returned. Otherwise arg itself is the text.

    Returns None and prints an error if the argument is empty or the file
    can't be read.  `usage_hint` is shown in the empty-argument error message.
    """
    arg = arg.strip()

    if arg.startswith("-f "):
        filepath = arg[3:].strip()
        if not filepath:
            console.print(f"[red]{usage_hint}[/red]")
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                text = fh.read()
        except FileNotFoundError:
            console.print(f"[red]File not found: {filepath!r}[/red]")
            return None
        except OSError as e:
            console.print(f"[red]Could not read file: {e}[/red]")
            return None
        label = f"[dim](from file: {filepath})[/dim]"
    else:
        text = arg
        label = ""

    if not text.strip():
        console.print(f"[red]{usage_hint}[/red]")
        return None

    return text, label


def extract_flag(args: str, flag: str) -> tuple[str, str | None]:
    """
    Find and remove a `flag <value>` pair from anywhere in `args`.

    Returns (remaining_args, value) if found, or (args, None) if not present.

    Example:
        extract_flag("hello world -o out.txt", "-o")
        => ("hello world", "out.txt")
    """
    tokens = args.split()
    if flag not in tokens:
        return args, None

    idx = tokens.index(flag)
    if idx + 1 >= len(tokens):
        # Flag present but no value after it.
        console.print(f"[red]{flag} requires a filepath argument[/red]")
        # Return the args unchanged so the caller can still show usage.
        return args, None

    value = tokens[idx + 1]
    remaining_tokens = tokens[:idx] + tokens[idx + 2:]
    return " ".join(remaining_tokens), value


def write_output(filepath: str, content: str):
    """Write `content` to `filepath` (UTF-8) and print a confirmation."""
    try:
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(content)
        console.print(f"[green]Output written to {filepath!r}[/green]")
    except OSError as e:
        console.print(f"[red]Could not write to {filepath!r}: {e}[/red]")


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_analyze(args: str, state: dict):
    result = load_text(
        args,
        "Usage: analyze <text>\n       analyze -f <filepath>",
    )
    if result is None:
        return
    text, label = result

    preview = text[:80].replace("\n", " ") + ("..." if len(text) > 80 else "")
    console.print(f"\nAnalyzing: [italic]{preview!r}[/italic] {label}")
    results = top_features_for_text(
        text, state["model"], state["sae"], state["config"].hook_point,
        k=10, device=state["device"],
    )

    table = Table(title=f'Top features for "{preview}"', show_lines=True)
    table.add_column("Rank",       style="dim",  width=6)
    table.add_column("Feature ID", style="cyan", width=12)
    table.add_column("Activation", style="green", width=12)
    table.add_column("Neuronpedia link", style="blue")

    for rank, (feat_id, val) in enumerate(results, 1):
        table.add_row(str(rank), str(feat_id), f"{val:.3f}", npedia(feat_id, state))

    console.print(table)
    console.print("[dim]Look up feature IDs at Neuronpedia to see human-written interpretations.[/dim]\n")


def cmd_feature(args: str, state: dict):
    """Show how strongly one feature activates at each token position."""
    # Split off the feature ID first; the rest is either "-f <path>" or inline text.
    parts = args.strip().split(None, 1)
    if len(parts) < 2:
        console.print("[red]Usage: feature <id> <text>[/red]")
        console.print("[red]       feature <id> -f <filepath>[/red]")
        return

    try:
        feat_id = int(parts[0])
    except ValueError:
        console.print(f"[red]Feature ID must be an integer, got: {parts[0]!r}[/red]")
        return

    result = load_text(
        parts[1],
        "Usage: feature <id> <text>\n       feature <id> -f <filepath>",
    )
    if result is None:
        return
    text, label = result

    if label:
        console.print(f"Text source: {label}")

    results = token_feature_activations(
        text, feat_id, state["model"], state["sae"],
        state["config"].hook_point, state["device"],
    )

    max_val = max((v for _, v in results), default=1.0)

    table = Table(title=f"Feature {feat_id} — per-token activation", show_lines=True)
    table.add_column("Pos",        style="dim",  width=5)
    table.add_column("Token",      style="cyan", width=16)
    table.add_column("Activation", style="green", width=12)
    table.add_column("Intensity",  width=25)

    for i, (token, val) in enumerate(results):
        # Simple ASCII bar proportional to this token's activation.
        bar_len = int((val / max(max_val, 1e-6)) * 20) if val > 0 else 0
        bar = "█" * bar_len
        if val > max_val * 0.7:
            bar_str = f"[bold green]{bar}[/bold green]"
        elif val > 0:
            bar_str = f"[green]{bar}[/green]"
        else:
            bar_str = "[dim]·[/dim]"

        table.add_row(str(i), repr(token), f"{val:.4f}", bar_str)

    console.print(table)
    console.print(f"[dim]Neuronpedia: {npedia(feat_id, state)}[/dim]\n")


def cmd_clamp(args: str, state: dict):
    """Add or update a feature clamp."""
    parts = args.strip().split()
    if not parts:
        console.print("[red]Usage: clamp <feature_id> [value][/red]")
        return

    try:
        feat_id = int(parts[0])
    except ValueError:
        console.print(f"[red]Feature ID must be an integer, got: {parts[0]!r}[/red]")
        return

    # Default value: 20.0 — a strong but not extreme activation.
    value = 20.0
    if len(parts) > 1:
        try:
            value = float(parts[1])
        except ValueError:
            console.print(f"[red]Value must be a number, got: {parts[1]!r}[/red]")
            return

    state["steerer"].clamp(feat_id, value)
    console.print(f"[green]Clamped feature {feat_id} = {value}[/green]")
    console.print(f"[dim]  {npedia(feat_id, state)}[/dim]\n")


def cmd_unclamp(args: str, state: dict):
    """Remove a single feature clamp."""
    parts = args.strip().split()
    if not parts:
        console.print("[red]Usage: unclamp <feature_id>[/red]")
        return

    try:
        feat_id = int(parts[0])
    except ValueError:
        console.print(f"[red]Feature ID must be an integer, got: {parts[0]!r}[/red]")
        return

    steerer = state["steerer"]
    if feat_id in steerer.clamped_features:
        steerer.unclamp(feat_id)
        console.print(f"[yellow]Feature {feat_id} released.[/yellow]\n")
    else:
        console.print(f"[dim]Feature {feat_id} was not clamped.[/dim]\n")


def cmd_list(state: dict):
    """Print the current clamp state."""
    clamps = state["steerer"].list_clamps()
    if not clamps:
        console.print("[dim]No features currently clamped.[/dim]\n")
        return

    table = Table(title="Active Feature Clamps", show_lines=True)
    table.add_column("Feature ID", style="cyan")
    table.add_column("Value",      style="green")
    table.add_column("Neuronpedia", style="blue")

    for feat_id, val in sorted(clamps.items()):
        table.add_row(str(feat_id), f"{val:.1f}", npedia(feat_id, state))

    console.print(table)
    console.print()


def _parse_n_flag(args: str, state: dict) -> tuple[str, int]:
    """Extract -n <N> from args, returning (remaining_args, max_new_tokens).
    Falls back to the current model's default_max_tokens if -n is not given."""
    args, n_str = extract_flag(args, "-n")
    if n_str is not None:
        try:
            return args, int(n_str)
        except ValueError:
            console.print(f"[red]-n requires an integer, got {n_str!r} — using model default[/red]")
    return args, state["config"].default_max_tokens


def cmd_generate(args: str, state: dict):
    """Generate text from a prompt using the current clamp configuration."""
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

    # For instruction-tuned models, wrap the prompt in the chat format so the
    # model knows it's a user message and should produce an assistant reply.
    formatted_prompt = format_prompt(prompt, state["config"])
    if formatted_prompt != prompt:
        console.print(f"[dim](chat format applied for {state['config'].display_name})[/dim]")

    steerer = state["steerer"]
    clamps = steerer.list_clamps()
    status = f"Generating with {len(clamps)} clamped feature(s)..." if clamps else "Generating (no features clamped)..."
    console.print(
        f"[bold]{status}[/bold]"
        + (f" {src_label}" if src_label else "")
        + f" [dim](max {max_new_tokens} tokens)[/dim]"
    )

    generated = steerer.generate(formatted_prompt, max_new_tokens=max_new_tokens, verbose=False)

    console.print(Panel(
        f"[bold blue]Prompt:[/bold blue]\n{prompt}\n\n[bold green]Generated:[/bold green]\n{generated}",
        title="Output",
        expand=False,
    ))
    console.print()

    if outpath:
        clamp_summary = ", ".join(f"{fid}={val}" for fid, val in sorted(clamps.items())) or "none"
        write_output(outpath, (
            f"=== generate output ===\n"
            f"model: {state['config'].display_name}\n"
            f"max_new_tokens: {max_new_tokens}\n"
            f"clamped features: {clamp_summary}\n\n"
            f"--- prompt ---\n{prompt}\n\n"
            f"--- generated ---\n{generated}\n"
        ))


def cmd_compare(args: str, state: dict):
    """
    Generate from the same prompt twice: once normally, once with clamps.
    Lets you directly compare the effect of your feature clamps.
    """
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
        console.print(f"Prompt source: {src_label}")

    formatted_prompt = format_prompt(prompt, state["config"])
    if formatted_prompt != prompt:
        console.print(f"[dim](chat format applied for {state['config'].display_name})[/dim]")

    steerer = state["steerer"]
    clamps = steerer.list_clamps()
    if not clamps:
        console.print("[yellow]No features clamped — both outputs will be identical.[/yellow]")
    console.print(f"[dim](max {max_new_tokens} tokens per generation)[/dim]")

    console.print("[bold]Generating normal output...[/bold]")
    steerer.unclamp_all()
    normal_out = steerer.generate(formatted_prompt, max_new_tokens=max_new_tokens, verbose=False)

    # Restore the clamps.
    for feat_id, val in clamps.items():
        steerer.clamp(feat_id, val)

    console.print("[bold]Generating steered output...[/bold]")
    steered_out = steerer.generate(formatted_prompt, max_new_tokens=max_new_tokens, verbose=False)

    left  = Panel(f"[bold]Prompt:[/bold]\n{prompt}\n\n[bold]Output:[/bold]\n{normal_out}",
                  title="Normal (no clamps)", border_style="blue")
    right = Panel(f"[bold]Prompt:[/bold]\n{prompt}\n\n[bold]Output:[/bold]\n{steered_out}",
                  title=f"Steered ({len(clamps)} clamp(s))", border_style="green")

    console.print(Columns([left, right], equal=True))
    console.print()

    if outpath:
        clamp_summary = ", ".join(f"{fid}={val}" for fid, val in sorted(clamps.items())) or "none"
        write_output(outpath, (
            f"=== compare output ===\n"
            f"model: {state['config'].display_name}\n"
            f"max_new_tokens: {max_new_tokens}\n"
            f"clamped features: {clamp_summary}\n\n"
            f"--- prompt ---\n{prompt}\n\n"
            f"--- normal (no clamps) ---\n{normal_out}\n\n"
            f"--- steered ({len(clamps)} clamp(s)) ---\n{steered_out}\n"
        ))


def cmd_models(_args: str, _state: dict):
    """List all available models."""
    table = Table(title="Available Models", show_lines=True)
    table.add_column("Key",         style="cyan",  width=16)
    table.add_column("Description", style="white")
    table.add_column("Chat?",       style="green", width=7)

    for key, cfg in MODELS.items():
        table.add_row(key, cfg.display_name, "yes" if cfg.is_chat else "no")

    console.print(table)
    console.print("[dim]Switch with: model <key>[/dim]\n")


def cmd_model(args: str, state: dict):
    """
    Switch to a different model+SAE. Loads everything fresh and resets all clamps.

    Example:  model gemma-2b-it
    """
    key = args.strip()
    if not key:
        current = state["config_name"]
        console.print(f"[dim]Current model: [cyan]{current}[/cyan]. Use 'models' to list options.[/dim]\n")
        return

    if key not in MODELS:
        known = ", ".join(f'[cyan]{k}[/cyan]' for k in MODELS)
        console.print(f"[red]Unknown model {key!r}. Available: {known}[/red]\n")
        return

    if key == state["config_name"]:
        console.print(f"[dim]Already using {key!r}.[/dim]\n")
        return

    console.print(f"[bold]Switching to {MODELS[key].display_name}...[/bold]")
    console.print("[dim]This will download weights on first use.[/dim]")

    # Load the new model FIRST. Only free the old one after a successful load
    # so that a failed load (auth error, network issue, etc.) leaves the
    # current model intact and the REPL stays usable.
    import gc
    model, sae, config = load_model_and_sae(key, device=state["device"])

    # Swap: discard old model now that the new one is ready.
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

    console.print(f"[bold green]Switched to {config.display_name}[/bold green]\n")


def print_help():
    console.print(Panel(
        "\n"
        "  [cyan]analyze[/cyan] [italic]<text>[/italic]              — show top 10 features activated by text\n"
        "  [cyan]analyze -f[/cyan] [italic]<path>[/italic]           — same, read text from a file\n"
        "  [cyan]feature[/cyan] [italic]<id> <text>[/italic]         — show per-token activation of a feature\n"
        "  [cyan]feature[/cyan] [italic]<id> -f <path>[/italic]      — same, read text from a file\n"
        "  [cyan]clamp[/cyan] [italic]<id> [value][/italic]          — force feature to value (default: 20.0)\n"
        "  [cyan]unclamp[/cyan] [italic]<id>[/italic]                — remove clamp on one feature\n"
        "  [cyan]clear[/cyan]                       — remove all clamps\n"
        "  [cyan]list[/cyan]                        — show currently clamped features\n"
        "  [cyan]generate[/cyan] [italic]<prompt>[/italic]                — generate text with current clamps\n"
        "  [cyan]generate -f[/cyan] [italic]<path>[/italic]             — same, read prompt from a file\n"
        "  [cyan]generate[/cyan] [italic]... -o <path>[/italic]         — write output to a file\n"
        "  [cyan]compare[/cyan] [italic]<prompt>[/italic]               — normal vs. steered output side by side\n"
        "  [cyan]compare -f[/cyan] [italic]<path>[/italic]              — same, read prompt from a file\n"
        "  [cyan]compare[/cyan] [italic]... -o <path>[/italic]          — write both outputs to a file\n"
        "  [cyan]models[/cyan]                      — list available models\n"
        "  [cyan]model[/cyan] [italic]<key>[/italic]               — switch to a different model\n"
        "  [cyan]rpt[/cyan]                         — repeat the last command\n"
        "  [cyan]help[/cyan]                        — show this message\n"
        "  [cyan]quit[/cyan]                        — exit\n"
        "\n"
        "  [bold]Workflow:[/bold]\n"
        "    1. [cyan]analyze[/cyan] some text to find interesting feature IDs\n"
        "    2. Look them up at neuronpedia.org to understand what they represent\n"
        "    3. [cyan]clamp[/cyan] a feature and [cyan]generate[/cyan] to see its effect\n"
        "    4. [cyan]compare[/cyan] to see normal vs. steered output side by side\n"
        "    5. [cyan]clear[/cyan] and try something different\n",
        title="Feature Steering Explorer — Help",
        expand=False,
    ))


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(Panel(
        "[bold]LLM Feature Steering Explorer[/bold]\n"
        "Inspired by Anthropic's Golden Gate Claude experiment.\n\n"
        f"Device: [cyan]{device}[/cyan]   |   "
        "Type [cyan]help[/cyan] for commands, [cyan]models[/cyan] to list models, [cyan]quit[/cyan] to exit.",
        expand=False,
    ))
    console.print()

    # Load the default model (GPT-2 small). Use 'model <key>' to switch.
    model, sae, config = load_model_and_sae(DEFAULT_MODEL, device=device)

    # All mutable REPL state lives in one dict so commands can update it
    # (e.g. when switching models) and the closures always see the new values.
    state = {
        "model":       model,
        "sae":         sae,
        "config":      config,
        "config_name": DEFAULT_MODEL,
        "steerer":     FeatureSteerer(model, sae, config.hook_point, device=device),
        "device":      device,
    }

    console.print("[bold green]Ready![/bold green] Type [cyan]help[/cyan] to see commands.\n")

    # All commands receive `state` so they always operate on the current model.
    COMMANDS = {
        "analyze":  lambda args: cmd_analyze(args, state),
        "feature":  lambda args: cmd_feature(args, state),
        "clamp":    lambda args: cmd_clamp(args, state),
        "unclamp":  lambda args: cmd_unclamp(args, state),
        "clear":    lambda _:    (state["steerer"].unclamp_all(),
                                  console.print("[yellow]All clamps removed.[/yellow]\n")),
        "list":     lambda _:    cmd_list(state),
        "generate": lambda args: cmd_generate(args, state),
        "compare":  lambda args: cmd_compare(args, state),
        "models":   lambda args: cmd_models(args, state),
        "model":    lambda args: cmd_model(args, state),
        "help":     lambda _:    print_help(),
    }

    last_cmd, last_args = None, ""   # track last command for `rpt`

    while True:
        try:
            raw = console.input("[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd  = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        elif cmd in ("rpt", "repeat"):
            if last_cmd is None:
                console.print("[dim]Nothing to repeat yet.[/dim]\n")
                continue
            console.print(f"[dim]Repeating: {last_cmd} {last_args}[/dim]")
            cmd, args = last_cmd, last_args
            # fall through to execute below

        if cmd in COMMANDS:
            try:
                COMMANDS[cmd](args)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]\n")
            # Only update last_cmd for real commands, not rpt itself.
            last_cmd, last_args = cmd, args
        elif cmd not in ("rpt", "repeat"):
            console.print(
                f"[red]Unknown command: {cmd!r}[/red] — type [cyan]help[/cyan] for a list.\n"
            )


if __name__ == "__main__":
    main()

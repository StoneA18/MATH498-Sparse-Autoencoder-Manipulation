"""
demo.py — Non-interactive demonstration of feature analysis and steering.

Run with:  python demo.py

This script walks through the core concepts in a fixed sequence so you can
verify everything works and get a concrete feel for the tools before using
the interactive explore.py.

What it demonstrates:
  1. Feature analysis  — which features activate for a given phrase
  2. Token breakdown   — where in the text a feature fires most
  3. Baseline generation — what GPT-2 says normally
  4. Steered generation  — what it says with a feature clamped
"""

import torch
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from feature_steering import (
    load_model_and_sae,
    top_features_for_text,
    token_feature_activations,
    FeatureSteerer,
    TARGET_LAYER,
)

console = Console()

NEURONPEDIA = "https://neuronpedia.org/gpt2-small/{layer}-res-jb/{fid}"


def section(title: str):
    console.print()
    console.print(Rule(f"[bold]{title}[/bold]", style="cyan"))
    console.print()


def neuronpedia_url(fid: int) -> str:
    return NEURONPEDIA.format(layer=TARGET_LAYER, fid=fid)


def demo_feature_analysis(model, sae, hook_point, device):
    """
    Demo 1: Feature Analysis
    Show which SAE features activate most strongly for a given phrase.
    """
    section("Demo 1 — Feature Analysis")

    examples = [
        "The Eiffel Tower stands in the heart of Paris.",
        "She opened the hood and checked the engine oil.",
        "The patient was diagnosed with type 2 diabetes.",
    ]

    for text in examples:
        console.print(f"[bold]Text:[/bold] {text!r}")
        features = top_features_for_text(text, model, sae, hook_point, k=5, device=device)

        for rank, (fid, val) in enumerate(features, 1):
            bar = "█" * int(val)
            console.print(
                f"  #{rank:1d}  feature [cyan]{fid:6d}[/cyan]  "
                f"activation [green]{val:6.2f}[/green]  "
                f"[dim]{neuronpedia_url(fid)}[/dim]"
            )
        console.print()

    console.print(
        "[dim]Tip: Copy any feature ID into Neuronpedia to read its human-written interpretation.[/dim]"
    )


def demo_token_breakdown(model, sae, hook_point, device):
    """
    Demo 2: Token-Level Feature Breakdown
    Pick a specific feature and see which tokens in a phrase activate it most.
    This builds intuition about what a feature actually 'detects'.
    """
    section("Demo 2 — Token-Level Feature Breakdown")

    text = "The capital of France is Paris, a city famous for the Eiffel Tower."

    console.print(f"[bold]Text:[/bold] {text!r}")
    console.print()

    # First find the top feature so we always look at a relevant one.
    top = top_features_for_text(text, model, sae, hook_point, k=1, device=device)
    feature_id, peak = top[0]

    console.print(
        f"Top feature: [cyan]{feature_id}[/cyan]  "
        f"(peak activation: [green]{peak:.2f}[/green])\n"
        f"[dim]{neuronpedia_url(feature_id)}[/dim]\n"
    )

    results = token_feature_activations(text, feature_id, model, sae, hook_point, device)
    max_val = max((v for _, v in results), default=1.0)

    for tok, val in results:
        bar_len = int((val / max(max_val, 1e-6)) * 30) if val > 0 else 0
        bar = "█" * bar_len
        intensity = "bold green" if val > max_val * 0.6 else ("green" if val > 0 else "dim")
        console.print(
            f"  {repr(tok):18s}  {val:6.4f}  [{intensity}]{bar}[/{intensity}]"
        )


def demo_steering(model, sae, hook_point, device):
    """
    Demo 3: Feature Steering
    Clamp a feature and compare the model's output to the un-clamped baseline.

    We search among the top features for text about Paris and find one that fires
    on a content token (not the BOS token TransformerLens prepends). Then we clamp
    it at 1.5x its natural peak — strong enough to see an effect without producing
    complete gibberish.

    Note: features that fire almost entirely at position 0 (BOS) are structural
    features, not semantic ones. Amplifying those heavily breaks generation.
    Use token_feature_activations() to check where a feature fires before clamping.
    """
    section("Demo 3 — Feature Steering (Clamping)")

    steerer = FeatureSteerer(model, sae, hook_point, device=device)

    # Find a feature that fires on content tokens (not just the BOS token).
    probe_text = "The Eiffel Tower is located in Paris, France."
    top_features = top_features_for_text(probe_text, model, sae, hook_point, k=10, device=device)

    # Default to the top feature; override if we find one that fires on content.
    feature_id, peak_val = top_features[0]
    for fid, fval in top_features:
        breakdown = token_feature_activations(probe_text, fid, model, sae, hook_point, device)
        peak_pos = max(range(len(breakdown)), key=lambda i: breakdown[i][1])
        if peak_pos > 0:  # fires on a content token, not position-0 BOS
            feature_id, peak_val = fid, fval
            break

    # 1.5x peak: noticeable steering without total incoherence.
    clamp_value = peak_val * 1.5

    console.print(f"Probe text: [italic]{probe_text!r}[/italic]")
    console.print(
        f"Selected feature: [cyan]{feature_id}[/cyan]  "
        f"natural peak = [green]{peak_val:.2f}[/green]  "
        f"-> clamping to [bold green]{clamp_value:.2f}[/bold green]"
    )
    console.print(f"[dim]{neuronpedia_url(feature_id)}[/dim]\n")

    prompt = "I want to tell you a little about myself."

    # --- Baseline (no steering) ---
    console.print("[bold]Generating baseline (no clamp)...[/bold]")
    baseline = steerer.generate(prompt, max_new_tokens=80, verbose=False)
    console.print(Panel(
        f"[bold blue]Prompt:[/bold blue] {prompt}\n\n"
        f"[bold]Continuation:[/bold]\n{baseline}",
        title="Baseline output",
        border_style="blue",
    ))

    # --- Steered (feature clamped) ---
    steerer.clamp(feature_id, clamp_value)
    console.print(f"\n[bold]Generating steered (feature {feature_id} = {clamp_value:.1f})...[/bold]")
    steered = steerer.generate(prompt, max_new_tokens=80, verbose=False)
    console.print(Panel(
        f"[bold blue]Prompt:[/bold blue] {prompt}\n\n"
        f"[bold]Continuation:[/bold]\n{steered}",
        title=f"Steered output (feature {feature_id} clamped to {clamp_value:.1f})",
        border_style="green",
    ))
    steerer.unclamp_all()

    console.print()
    console.print(
        "[dim]The steered output should show the clamped concept bleeding into the response.\n"
        "Effect strength depends on which feature fired — some are more 'steerable' than others.[/dim]"
    )


def demo_suppress(model, sae, hook_point, device):
    """
    Demo 4: Feature Suppression
    Clamping a feature to 0 suppresses it even when it would normally be active.
    This is the opposite of amplification — useful for ablation experiments.
    """
    section("Demo 4 — Feature Suppression (Clamping to Zero)")

    steerer = FeatureSteerer(model, sae, hook_point, device=device)

    # Pick text about a specific topic and suppress its top feature.
    topic_text = "The doctor examined the patient carefully."
    top = top_features_for_text(topic_text, model, sae, hook_point, k=1, device=device)
    feature_id, peak_val = top[0]

    console.print(f"Topic text: [italic]{topic_text!r}[/italic]")
    console.print(
        f"Top feature: [cyan]{feature_id}[/cyan]  "
        f"natural peak = [green]{peak_val:.2f}[/green]  "
        f"→  suppressing to [bold red]0.0[/bold red]"
    )
    console.print(f"[dim]{neuronpedia_url(feature_id)}[/dim]\n")

    prompt = "The doctor walked into the room and"

    console.print("[bold]Baseline output (no suppression):[/bold]")
    baseline = steerer.generate(prompt, max_new_tokens=60, verbose=False)
    console.print(f"  {prompt}[green]{baseline}[/green]\n")

    steerer.clamp(feature_id, 0.0)
    console.print(f"[bold]Suppressed output (feature {feature_id} = 0.0):[/bold]")
    suppressed = steerer.generate(prompt, max_new_tokens=60, verbose=False)
    console.print(f"  {prompt}[yellow]{suppressed}[/yellow]\n")
    steerer.unclamp_all()

    console.print(
        "[dim]Suppression results vary — GPT-2 small has limited semantic richness.\n"
        "The effect is much stronger in larger models with more specialized features.[/dim]"
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(Panel(
        "[bold]GPT-2 Feature Steering — Demo Script[/bold]\n\n"
        "This script runs through four demonstrations automatically.\n"
        "First run will download model weights (~500 MB).\n\n"
        f"Device: [cyan]{device}[/cyan]",
        expand=False,
    ))

    model, sae, hook_point = load_model_and_sae(device=device)

    demo_feature_analysis(model, sae, hook_point, device)
    demo_token_breakdown(model, sae, hook_point, device)
    demo_steering(model, sae, hook_point, device)
    demo_suppress(model, sae, hook_point, device)

    section("Done")
    console.print(
        "Explore interactively with [bold cyan]python explore.py[/bold cyan]\n"
    )


if __name__ == "__main__":
    main()

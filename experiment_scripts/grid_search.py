"""
Grid search over CondDistOp parameters (p × mean × std) for a single feature.

CLI (via experiments.py):
    uv run experiments.py grid-search feature=18493 p=0,0.8,0.1 dist=normal mean=0,60,5 std=0,30,10 prompt='The president is most concerned about the threat of' n=250

Range syntax: start,stop,step  — stop is exclusive (numpy arange semantics).
All arguments are plain key=value with no parentheses, so no shell quoting is needed.
"""

from pathlib import Path
import numpy as np


def _parse_range(s: str) -> list[float]:
    """'start,stop,step' → list of floats via np.arange (stop exclusive)."""
    start, stop, step = (float(x.strip()) for x in s.split(","))
    return [round(float(v), 10) for v in np.arange(start, stop, step)]


def _is_range(s: str) -> bool:
    """True when s is exactly three comma-separated numbers."""
    parts = s.split(",")
    if len(parts) != 3:
        return False
    try:
        [float(p) for p in parts]
        return True
    except ValueError:
        return False


def parse_grid_search_args(raw_args: list[str]) -> dict:
    """Parse grid-search CLI tokens into a kwargs dict for run_grid_search.

    Any value that is exactly 'start,stop,step' (three comma-separated numbers)
    is treated as a range spec and expanded via np.arange.  Everything else is
    coerced to int, float, or kept as a string.
    """
    result: dict = {}
    for arg in raw_args:
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        key, val = key.strip(), val.strip()
        if _is_range(val):
            result[key] = _parse_range(val)
        else:
            try:
                result[key] = int(val)
            except ValueError:
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
    return result


def run_grid_search(
    feature: int = 18493,
    p=None,
    dist: str = "normal",
    mean=None,
    std=None,
    prompt: str = "The president is most concerned about the threat of",
    n: int = 250,
):
    """Sweep p × mean × std, write one .txt per combination to experiment_data/grid_search/."""
    from steering_utils import load_gpt2_small, load_sae_from_neuronpedia, Dist

    feature = int(feature)
    n = int(n)

    p_vals = p    if isinstance(p, list)    else ([float(p)]    if p    is not None else [0.5])
    means  = mean if isinstance(mean, list) else ([float(mean)] if mean is not None else [40.0])
    stds   = std  if isinstance(std, list)  else ([float(std)]  if std  is not None else [10.0])

    output_dir = Path("experiment_scripts/experiment_data/grid_search")
    output_dir.mkdir(parents=True, exist_ok=True)

    combos = [(pv, mv, sv) for pv in p_vals for mv in means for sv in stds]
    total  = len(combos)

    print(f"Grid search: {len(p_vals)} p × {len(means)} mean × {len(stds)} std = {total} combinations")
    print("Loading model and SAE...")

    model = load_gpt2_small()
    sae   = load_sae_from_neuronpedia("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    model.add_sae(sae)

    print(f'Prompt: "{prompt}"\n')

    for idx, (pv, mv, sv) in enumerate(combos, 1):
        sae.clear()

        # std=0 is degenerate for Normal; use tiny epsilon so Dist doesn't error
        actual_std = max(float(sv), 1e-6)
        sae.cond_dist(feature, float(pv), Dist(dist, float(mv), actual_std))

        full_text  = model.generate(prompt, max_tokens=n, log=False)
        completion = full_text[len(prompt):]

        fname = f"f{feature}_p{pv:.3f}_mean{mv:.1f}_std{sv:.1f}.txt"
        fpath = output_dir / fname

        header = (
            f"feature={feature} p={pv:.4f} dist={dist} "
            f"mean={mv:.2f} std={sv:.2f} n={n} "
            f'prompt="{prompt}"'
        )
        fpath.write_text(header + "\n" + completion.strip(), encoding="utf-8")

        print(f"({idx}/{total}) {fname}", flush=True)

    sae.clear()
    print(f"\nDone — {total} files written to {output_dir}")

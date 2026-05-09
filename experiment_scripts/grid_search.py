"""
Grid search over CondDistOp parameters (p × mean × std) for a single feature.

Run the sweep:
    uv run experiments.py grid-search feature=18493 p=0,0.8,0.1 dist=normal mean=0,60,5 std=0,30,10 prompt='The president is most concerned about the threat of' n=250

Score the results (requires AGENT_KEY env var set to an OpenAI API key):
    uv run experiments.py grid-search process-results

Range syntax: start,stop,step  — stop is exclusive (numpy arange semantics).
All arguments are plain key=value with no parentheses, so no shell quoting is needed.
"""

import csv
import os
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Shared paths
# ---------------------------------------------------------------------------

DATA_DIR   = Path("experiment_scripts/experiment_data/grid_search")
SCORES_CSV = DATA_DIR / "assessed_scores.csv"


# ---------------------------------------------------------------------------
# Argument parsing (sweep command)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

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

    DATA_DIR.mkdir(parents=True, exist_ok=True)

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
        fpath = DATA_DIR / fname

        header = (
            f"feature={feature} p={pv:.4f} dist={dist} "
            f"mean={mv:.2f} std={sv:.2f} n={n} "
            f'prompt="{prompt}"'
        )
        fpath.write_text(header + "\n" + completion.strip(), encoding="utf-8")

        print(f"({idx}/{total}) {fname}", flush=True)

    sae.clear()
    print(f"\nDone — {total} files written to {DATA_DIR}")


# ---------------------------------------------------------------------------
# Evaluation prompts
# ---------------------------------------------------------------------------

_RELATABILITY_PROMPT = """\
<task>
As an evaluator in this task, your responsibility is to assess how naturally a language model completed a given prompt.
Your objective is to determine whether the model's response is a plausible and on-topic continuation of the provided prompt — as if a knowledgeable human had written it.
You return an integer score, ranging from 0 (the completion is completely implausible, off-topic, or ignores the prompt entirely) to 100 (the response reads as a perfectly natural, on-topic completion of the prompt).
Each task will present the prompt given to the model and the response it generated.
Here is an example:
<prompt>
The biggest challenge facing renewable energy is
</prompt>
<response>
Bananas grow in tropical climates and require significant water.
</response>
For the above example, the score would be:
2
(This very low score reflects that the response has nothing to do with the prompt.)
Only return an integer score between 0 and 100. *Do not* include any other text in your response EVER, besides the integer score.
</task>
<prompt>
{prompt}
</prompt>
<response>
{response}
</response>"""

_SHARK_STRENGTH_PROMPT = """\
<task>
As an evaluator in this task, your responsibility is to assess how strongly the content of a language model's response relates to the concept of sharks.
You return an integer score, ranging from 0 (the response contains no reference to or association with sharks whatsoever) to 100 (the response is entirely and centrally about sharks).
Consider direct mentions of sharks, shark-related topics (species, behaviour, habitat, attacks, fins, teeth, oceans, etc.), and indirect associations.
Here is an example:
<response>
The great white shark is the most feared predator in the ocean, known for its powerful jaws and rows of serrated teeth.
</response>
For the above example, the score would be:
97
(This very high score reflects that the response is entirely focused on sharks.)
Only return an integer score between 0 and 100. *Do not* include any other text in your response EVER, besides the integer score.
</task>
<response>
{response}
</response>"""

_COHERENCY_PROMPT = """\
<task>
As an evaluator in this task, your responsibility is to assess how coherent and fluent a language model's response is.
Your objective is to determine whether the text is grammatically sound, logically consistent within itself, and reads as natural prose rather than garbled or incoherent output.
You return an integer score, ranging from 0 (completely incoherent, nonsensical, or garbled) to 100 (perfectly coherent, fluent, and well-structured).
Do not penalise unusual or surprising content — only evaluate fluency and internal logical consistency.
Here is an example:
<response>
The the the moon is cheese purple running backwards always never.
</response>
For the above example, the score would be:
3
(This very low score reflects that the text is grammatically and logically broken.)
Only return an integer score between 0 and 100. *Do not* include any other text in your response EVER, besides the integer score.
</task>
<response>
{response}
</response>"""


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

_EVAL_MODEL = "gpt-4o-mini"

CSV_FIELDS = [
    "filename", "feature", "p", "dist", "mean", "std", "n", "prompt",
    "relatability", "shark_strength", "coherency",
]


def _parse_param_line(line: str) -> dict:
    """Parse the header line written by run_grid_search back into a dict."""
    params: dict = {}
    # Extract prompt="..." first (may contain spaces)
    m = re.search(r'prompt="([^"]*)"', line)
    if m:
        params["prompt"] = m.group(1)
        line = line[: m.start()].strip()
    for part in line.split():
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                params[k] = int(v)
            except ValueError:
                try:
                    params[k] = float(v)
                except ValueError:
                    params[k] = v
    return params


def _call_model(client, prompt_text: str) -> int:
    """Call the OpenAI eval model and return a 0-100 integer score."""
    resp = client.chat.completions.create(
        model=_EVAL_MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=5,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    # Extract first integer found in the response (guards against stray punctuation)
    m = re.search(r"\d+", raw)
    if not m:
        raise ValueError(f"Model returned non-integer: {raw!r}")
    return min(100, max(0, int(m.group())))


# ---------------------------------------------------------------------------
# process-results entry point
# ---------------------------------------------------------------------------

def process_results() -> None:
    """Score every .txt file in DATA_DIR with three OpenAI eval calls each."""
    from openai import OpenAI

    api_key = os.environ.get("AGENT_KEY")
    if not api_key:
        raise RuntimeError("AGENT_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    txt_files = sorted(DATA_DIR.glob("f*.txt"))
    if not txt_files:
        print(f"No result files found in {DATA_DIR}. Run the grid search first.")
        return

    # Load already-processed filenames so we can resume interrupted runs
    done: set[str] = set()
    if SCORES_CSV.exists() and SCORES_CSV.stat().st_size > 0:
        with SCORES_CSV.open(encoding="utf-8") as fh:
            done = {row["filename"] for row in csv.DictReader(fh)}

    write_header = not SCORES_CSV.exists() or SCORES_CSV.stat().st_size == 0

    total = len(txt_files)
    print(f"Scoring {total} files with {_EVAL_MODEL} ({total - len(done)} remaining)...\n")

    with SCORES_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for i, fpath in enumerate(txt_files, 1):
            if fpath.name in done:
                print(f"({i}/{total}) {fpath.name} — already scored, skipping", flush=True)
                continue

            lines = fpath.read_text(encoding="utf-8").splitlines()
            params   = _parse_param_line(lines[0])
            response = "\n".join(lines[1:]).strip()
            prompt   = params.get("prompt", "")

            rel   = _call_model(client, _RELATABILITY_PROMPT.format(prompt=prompt, response=response))
            shark = _call_model(client, _SHARK_STRENGTH_PROMPT.format(response=response))
            coh   = _call_model(client, _COHERENCY_PROMPT.format(response=response))

            writer.writerow({
                "filename":      fpath.name,
                "feature":       params.get("feature", ""),
                "p":             params.get("p", ""),
                "dist":          params.get("dist", ""),
                "mean":          params.get("mean", ""),
                "std":           params.get("std", ""),
                "n":             params.get("n", ""),
                "prompt":        prompt,
                "relatability":  rel,
                "shark_strength": shark,
                "coherency":     coh,
            })
            fh.flush()

            print(f"({i}/{total}) {fpath.name}  rel={rel}  shark={shark}  coh={coh}", flush=True)

    print(f"\nDone — scores written to {SCORES_CSV}")

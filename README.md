# README: Sparse Autoencoder Feature Steering

This repository contains code and interfaces for loading pretrained language models and sparse autoencoders, applying feature-level interventions during generation, and studying those interventions through interactive dashboards.
Built as part of MATH498 - Decoding GPT. Colorado School of Mines, Spring 2026.

**Authors:** Andy Holmberg, Stone Amsbaugh  
**Instructor:** Michael Ivanitsky

---

## Project Writeup
**Important:** This README is for those wishing to use the codebase that this repository contains. It is NOT a formal outline of our project and findings.

We **highly** recommend that you instead start by looking at our [writeup](docs/final_writeup/html/writeup.html). It will explain nicely what we have done and what we have found. Then, come back here to play with our code.

This is especially recommended if your name is Michael, or you are otherwise interested in assessing our work.

Enjoy!

---

## Table of Contents

- [Repository Contents](#repository-contents)
- [Setup](#setup)
- [steering_utils.py](#steering_utilspy)
  - [Loading Models and SAEs](#loading-models-and-saes)
  - [Clamping and Querying](#clamping-and-querying)
  - [Steering Operations](#steering-operations)
  - [Other Utilities](#other-utilities)
- [Experiments](#experiments)
  - [sample](#sample)
  - [dashboard_experiment](#dashboard_experiment)
  - [bulk_feature_stats](#bulk_feature_stats)
  - [affectation_experiment](#affectation_experiment)
  - [affectation_experiment_v2](#affectation_experiment_v2)
  - [grid-search](#grid-search)
- [Web](#web)
  - [SAE Playground](#sae-playground)
  - [Feature Steering Experiment Charts](#feature-steering-experiment-charts)

---

## Repository Contents

* **steering_utils.py** — The core utility file. All classes and functions for loading models and SAEs, applying steering interventions, generating text, and more are here.
* **trainable_sae.py** — Similar to the above, but instead contains the architecture and configuration classes for custom trainable SAEs, used by the playground server and training scripts.
* **sae_generation.py** — Exposes the playground's saved-vector generation path as a standalone Python API; covers the hot/cold and happy/sad preset delta vectors with additive or projection steering.
* **experiments.py** — A program used for quickly running experiments, it parses command line arguments to the experiment scripts themselves.
* **experiment_scripts/** — Individual experiment modules; each defines a function that is registered in `experiments.py`.
* **experiment_scripts/experiment_data/** — Training text files and output archives produced by experiments.
* **scripts/** — Utility scripts for data collection, SAE training, and keeping the web dashboards in sync with experiment output.
* **web/** — Browser-based dashboards and the live SAE playground server.
* **docs/** — Project documentation. `docs/final_writeup/` holds the final paper as both rendered HTML (with embedded interactive dashboards) and LaTeX source; `docs/old_updates/` holds earlier progress reports and the original project proposal.
* **samples/** — Sample prompt text files for querying or training models.
* **notebooks/** — Jupyter notebooks for interactive exploration.
* **pyproject.toml**, **uv.lock** — Package manager files; contain all project dependencies.
* **requirements.txt** — pip-compatible dependency list; use this if you are not using uv.
* **index.html** - Landing page for github pages.
* **docs/final_writeup/html/writeup.html** — The best place to start: a self-contained HTML page with the full writeup and all interactive result dashboards embedded.

**Note:** There may be other directories in your local repository, such as `outputs/` which logs model responses, as well as your trained or downloaded SAEs, that are not present in this public repository.

---

## Setup

We recommend running this project with uv. To install uv, see: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

Install dependencies:
```
uv sync
```

Then, run any experiment (Read the documentation for the experiment first) with:
```
uv run experiments.py <experiment_name> [key value ...]
```

Start the web playground server (Again, read the documentation first) with:
```
uv run python web/web_sae_playground.py --host 127.0.0.1 --port 8000
```

If you are experimenting with `steering_utils` directly, it is recommended you do so in a standalone script run with `uv run`.

---

## steering_utils.py

This is the main file you interact with when using the project programmatically. It defines the `SteerableModel`, `SteerableSAE`, `SteeringOp`, and `Dist` classes, as well as factory functions for loading models and SAEs.

### Loading Models and SAEs

Load a model with one of the two factory functions:
```python
from steering_utils import load_gpt2_small, load_gemma_3_270m_it

model = load_gpt2_small()       # GPT-2-small via TransformerLens
model = load_gemma_3_270m_it()  # Gemma-3-270m-IT
```
Both return a `SteerableModel` wrapping a TransformerLens `HookedTransformer`. Models are downloaded automatically on first use.

Load a SAE from Neuronpedia and attach it to the model:
```python
from steering_utils import load_sae_from_neuronpedia

sae = load_sae_from_neuronpedia("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
model.add_sae(sae)
```
The `release` and `sae_id` arguments match Neuronpedia's naming conventions. The hook point is inferred from the SAE metadata and can be overridden with the `hook_name` keyword argument. Multiple SAEs can be attached by calling `add_sae` multiple times.

### Clamping and Querying

Clamp a feature to a fixed activation value for all tokens on every forward pass:
```python
sae.clamp(18493, 40.0)
```

Remove all active interventions:
```python
sae.clear()
```

Generate text with any active clamps applied:
```python
result = model.generate("I like sharks", max_tokens=100)
```
Returns the full prompt + completion as a string. Each generation is logged to `output/log.txt` by default (`log=False` to suppress). Additional options: `from_file=True` to read the prompt from a path, `output_file` to also write the result to a file.

Start an interactive chat loop:
```python
model.chat(max_tokens=200)
```

Apply a conditional distribution instead of a hard clamp — with probability `p`, the feature activation is replaced by a sample from `dist`; otherwise it is left at its natural value:
```python
from steering_utils import Dist

sae.cond_dist(18493, p=0.3, dist=Dist("normal", 40, 10))
```

`Dist` supports: `normal`, `uniform`, `lognormal`, `exponential`, `beta`. Parameters are positional and follow standard parameterizations. `model.cond_dist(feature_id, p, dist)` applies the same distribution across all attached SAEs at once.

### Steering Operations

For the dashboard experiment, interventions are expressed as `SteeringOp` subclasses instead of the simpler clamp/cond_dist API. These are more expressive because they operate token-by-token. All ops are importable from `steering_utils`:

| Op | Behavior |
|---|---|
| `ClampOp(value)` | Always fix the feature to a constant value |
| `CondDistOp(p, dist)` | With probability `p`, replace with a sample from `dist` |
| `EveryOtherTokenOp(value)` | Clamp on alternating tokens |
| `NthTokenOp(n, value, offset=0)` | Clamp every n-th token |
| `SpecificTokensOp(indices, value)` | Clamp only at the listed absolute token positions |
| `FibonacciTokensOp(value)` | Clamp at Fibonacci-indexed positions (0, 1, 2, 3, 5, 8, 13, …) |
| `ThresholdOp(threshold, value)` | Clamp only when the natural activation exceeds `threshold` |
| `ScaleOp(scale)` | Multiply the activation by a scale factor |
| `AddOp(delta)` | Add a constant to the activation |
| `ZeroOp()` | Suppress the feature to zero |
| `ChainOp(other_feature, ratio)` | Tie this feature's activation to another feature's natural activation |

To define a new op, subclass `SteeringOp` and implement:
```python
def apply(self, current_val: torch.Tensor, token_idx: int) -> torch.Tensor
```
`current_val` is a shape-(1,) tensor for one feature at one token. `token_idx` is the absolute position in the sequence, including BOS.

### Other Utilities

**Feature analysis page:** Run a prompt through the SAE and produce an interactive HTML page showing the top activated features per token, color-coded by strength. Clicking a feature reveals its activation across all token positions. Includes a search bar.
```python
sae.analyze("I like sharks", html_output="output/analysis.html", top_n=10)
```

**Raw activations:** Collect the full `(n_tokens, n_features)` activation matrix for a body of text:
```python
matrix = sae.collect_activations("path/to/text.txt", from_file=True)
```

---

## Experiments

Experiments live in `experiment_scripts/` and are registered in `experiments.py`. Run any of them with:
```
uv run experiments.py <experiment_name> [key value ...]
```
Key-value arguments are parsed automatically and coerced to `int`, then `float`, then `str`. To list all available experiments:
```
uv run experiments.py
```

### sample

A minimal working example of loading a model and SAE, clamping a feature, generating text, and running `analyze`. Useful for verifying your environment works.
```
uv run experiments.py sample
```

### dashboard_experiment

Runs the `SteeringExperiment` pipeline: defines multiple steering methods for a single feature, generates text under each, and writes a self-contained HTML dashboard to `output/dashboard.html`.
```
uv run experiments.py dashboard_experiment
uv run experiments.py dashboard_experiment prompt "The ocean is" n_tokens 60 feature_id 18493
```
The dashboard lets you select between methods with a dropdown. Each token is colored gray (feature inactive), green (active, no intervention), or orange (intervention fired). Hovering shows a tooltip with natural and post-intervention activation values; a bolt marks where the op changed the value. A chart below shows activation over the full sequence, solid for post-op and dashed for natural. The output is also archived timestamped to `experiment_scripts/experiment_data/dashboard_experiment/`.

Parameters: `prompt` (default `"I like sharks"`), `n_tokens` (default `100`), `feature_id` (default `18493`), `output` (default `"output/dashboard.html"`).

### bulk_feature_stats

Passes a body of text through the GPT-2-small layer-8 SAE and collects per-feature activation statistics across all tokens. Writes `feature_stats.csv` with columns `feature_id`, `count`, `mean_when_active`, `std_when_active`, sorted by activation frequency. Useful for discovering which features are most active over a given corpus.
```
uv run experiments.py bulk_feature_stats
uv run experiments.py bulk_feature_stats text_path path/to/text.txt
```

### affectation_experiment

Compares SAE feature activations between a happy text corpus and a sad text corpus to identify emotionally-biased features. Has two modes:

**Mode 0** (default) — Loads `very_happy.txt` and `very_sad.txt` from `experiment_scripts/experiment_data/affectations/`, collects activations for each, and writes `affectation_stats.csv` with per-feature happy/sad counts and means.

**Mode 1** — Loads the existing stats CSV, clamps the top 10 most happy-biased features (highest happy − sad count) to their `mean_happy` activation value, and starts an interactive chat session with the steered model.
```
uv run experiments.py affectation_experiment
uv run experiments.py affectation_experiment mode 1
```

### affectation_experiment_v2

A more sophisticated version of the above using Gemma-3-270m-IT instead of GPT-2. Uses hot/cold word pairs (`hot`, `warm`, `blazing`, … vs. `cold`, `frigid`, `glacial`, …) randomly substituted into sentence templates to generate two contrasting corpora.

**Mode 0** (default) — Generates activations from hot and cold corpora, writes `affectations2_stats.csv`.

**Mode 1** — Loads the stats, clamps top hot-biased features, and generates three side-by-side completions on a fixed prompt: baseline, hard-clamped, and conditional-distribution. Results are appended with timestamps to `experiment_scripts/experiment_data/affectations2/responses.log`.
```
uv run experiments.py affectation_experiment_v2
uv run experiments.py affectation_experiment_v2 mode 1 top_n 10
```

### grid-search

Sweeps `CondDistOp` parameters (`p` × `mean` × `std`) for a single feature over a prompt and writes one output `.txt` file per parameter combination. Then, optionally scores every output with GPT-4o-mini on three dimensions: **relatability** (is it a natural continuation?), **shark strength** (how much does it reference sharks?), and **coherency** (is it grammatically sound?). Scores are written to `assessed_scores.csv`.

Run the sweep:
```
uv run experiments.py grid-search feature=18493 p=0,0.8,0.1 dist=normal mean=0,60,5 std=0,30,10 prompt="The president is most concerned about the threat of" n=250
```
Score the results (requires `AGENT_KEY` env var set to an OpenAI API key):
```
uv run experiments.py grid-search process-results
```
Range syntax is `start,stop,step` (numpy arange semantics, stop exclusive). The scoring step is resumable — already-scored files are tracked in the CSV and skipped on re-run.

After scoring, run `uv run scripts/update_plots.py` to bake the CSV into the analysis dashboard (see [Feature Steering Experiment Charts](#feature-steering-experiment-charts)).

---

## Web

### SAE Playground

The SAE playground is a live browser interface for experimenting with **custom trained SAEs**. It is a separate workflow from the Neuronpedia experiments above and is built around SAEs trained locally with `scripts/train_sae.py`.

**Start the server:**
```
uv run python web/web_sae_playground.py --host 127.0.0.1 --port 8000
```
An optional `--device` flag (e.g. `--device cuda:2`) controls which GPU/CPU is used. The server discovers all SAE checkpoints in `saved_saes/` automatically and lists them in a dropdown in the browser.

**Using the playground:** Select an SAE checkpoint and a steering mode. The primary mode is **notebook delta pair**: it computes a concept direction vector from two contrasting corpora (e.g. hot words vs. cold words) via either addition or projection. For a given prompt, it generates three outputs side by side — baseline (no intervention), positively steered (concept direction added), negatively steered (concept direction subtracted). Each run is logged to `web/log.csv`.

**Viewing archived runs:** Open `web/static_sae_playground.html` directly in a browser — no server required. This is a self-contained archive viewer with all past playground runs embedded in the page as JSON. Select any logged generation from the dropdown on the left to see the prompt, full settings metadata, and the three side-by-side outputs. The page is updated automatically by the server as new runs complete.

### Feature Steering Experiment Charts

`web/feature_steering_experiment_charts.html` is an interactive dashboard for visualizing grid search results. Open the file in a browser. You will see a file-upload drop zone — drag in an `assessed_scores.csv` to load the dashboard. Once loaded:

* **Metric cards** describe what each score means: Relatability, Shark Strength, and Coherency (each scored 0–100 by GPT-4o-mini).
* **Stat cards** show min, max, average, and standard deviation for each metric across all evaluated outputs.
* **Plotly charts** display scatter plots and parameter breakdowns for exploring how `p`, `mean`, and `std` affect each metric.
* A **raw data tab** shows the full CSV table.

To avoid uploading the CSV every time, bake it directly into the HTML:
```
uv run scripts/update_plots.py
```
This embeds the current `assessed_scores.csv` into the page as an inline JavaScript string so the dashboard loads pre-populated with no file upload or server needed.

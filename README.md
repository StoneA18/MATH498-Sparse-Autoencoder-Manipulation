# README: Sparse Autoencoder Feature Steering

This repository contains code and an interface for loading pretrained language models and sparse autoencoders, applying feature-level interventions during generation, and comparing those interventions through an interactive HTML dashboard.
Built as part of MATH498 - Decoding GPT. Colorado School of Mines, Spring 2026.

**Authors:** Andy Holmberg, Stone Amsbaugh  
**Instructor:** Michael Ivanitsky

---

## Table of Contents

- [Overview](#overview)
- [Repository Contents](#repository-contents)
- [Using the Project](#using-the-project)
  - [Environment Setup](#environment-setup)
  - [Loading a Model](#loading-a-model)
  - [Loading a SAE](#loading-a-sae)
  - [Steering a Feature](#steering-a-feature)
  - [Conditional Distributions](#conditional-distributions)
  - [Generating Text](#generating-text)
  - [Analyzing Features](#analyzing-features)
  - [Interactive Chat](#interactive-chat)
  - [Running Experiments](#running-experiments)
  - [The Steering Dashboard](#the-steering-dashboard)
- [Project Structure](#project-structure)
  - [Dist](#dist)
  - [SteeringOp](#steeringop)
  - [SteerableSAE](#steerablesae)
  - [SteerableModel](#steerablemodel)
  - [Factory Functions](#factory-functions)
  - [SteeringExperiment and ExperimentResults](#steeringexperiment-and-experimentresults)
  - [experiments.py](#experimentspy)
- [Evaluation](#evaluation)
  - [Results](#results)
  - [Design Choices, Challenges, Future Direction](#design-choices-challenges-future-direction)
    - [Challenges](#challenges)
    - [Design Choices](#design-choices)
    - [Future Work](#future-work)
  - [Collaboration](#collaboration)
- [Attribution](#attribution)

---

## Overview

In MATH498, we have studied sparse autoencoders (SAEs) as a tool for mechanistic interpretability of large language models. SAEs decompose the internal activations of a model into a sparse set of interpretable features. In this project, we load pretrained models and pretrained SAEs from Neuronpedia, then build an interface for intervening on specific SAE features during generation in order to study how those features influence model behavior. The central question is: if we force a feature to be active, or modify it in various structured ways, what happens to the model's output?

## Repository Contents

For the purposes of navigating this repository, consider the following contents:

* **README.md:** That is this document. It describes the project and documents its functionality. Take a look around.
* **steering_utils.py:** This is the core utility file. It defines all classes and functions for loading models and SAEs, applying steering interventions, generating text, and producing HTML feature analysis pages. All of the real logic lives here.
* **experiments.py:** This is the experiment runner. It imports experiment functions from `experiment_scripts/` and exposes them through a simple CLI dispatcher. Run experiments from here.
* **experiment_scripts/:** This directory contains individual experiment scripts. Each script defines one or more experiment functions that are registered in `experiments.py`. Current experiments include:
  * `bulk_features.py` — Collects activation statistics (count, mean, std) for every feature across a body of text.
  * `affectations.py` — Compares feature activations across happy vs. sad text corpora to find emotionally-biased features, then applies them as clamps in an interactive chat.
  * `affectations2.py` — A more advanced version of the above using Gemma-3, conditional distributions, and hot/cold word substitutions.
  * `steering_dashboard.py` — The main experiment for this project. Defines `SteeringExperiment` and `ExperimentResults`, which run multiple steering strategies on a single prompt and render the results as an interactive HTML dashboard. See [The Steering Dashboard](#the-steering-dashboard) for full details.
* **experiment_scripts/experiment_data/:** Data used by and produced by experiments. Includes corpus text files and archives of dashboard outputs.
* **output/:** Generated outputs from running experiments. Includes `log.txt`, an append-only JSON log of every generation, and any HTML files produced during a session.
* **samples/:** Sample prompt text files for use with experiments.
* **notebooks/:** Jupyter notebooks for interactive exploration.
* **trainable_sae.py:** Configuration classes for custom trainable SAEs. This is mostly a reference file and is not part of the main experiment workflow.
* **pyproject.toml**, **uv.lock:** Created by uv, the package manager used to develop this project. Contains all project dependencies.
* **depracated/:** Old scripts and notebooks saved as a reference for the development process. There is no need to look here.

## Using the Project

Here we will describe the high-level functionality of the project, which lives in `steering_utils.py` and `experiment_scripts/steering_dashboard.py`. This is what you need to use the interface without understanding the inner workings.

### Environment Setup

We recommend running this project with uv, but it is not required. If you opt not to use uv, ensure that your environment is set up with the packages listed as dependencies in `pyproject.toml`.

To install and get started with uv, see: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

You can run:
```
uv sync
```
to install the project dependencies.

To run a script, use:
```
uv run experiments.py <experiment_name>
```

If you are going to experiment with the following usage instructions, it is recommended you do so in a Jupyter notebook or a standalone script.

### Loading a Model

The project supports GPT-2-small and Gemma-3-270m-IT. Load one with:
```python
from steering_utils import load_gpt2_small, load_gemma_3_270m_it

model = load_gpt2_small()
# or
model = load_gemma_3_270m_it()
```
This returns a `SteerableModel` object that wraps a TransformerLens `HookedTransformer`. The model will be downloaded automatically on first use.

### Loading a SAE

SAEs are loaded from Neuronpedia using a release name and a SAE ID. Attach the SAE to your model with `add_sae`:
```python
from steering_utils import load_sae_from_neuronpedia

sae = load_sae_from_neuronpedia("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
model.add_sae(sae)
```
When you attach a SAE, the project will print the hook point it is attached to and a direct Neuronpedia link for browsing features.

#### Notes:
* The `release` and `sae_id` arguments correspond directly to Neuronpedia's release naming conventions. You can find these on the Neuronpedia website for any SAE you are interested in.
* Multiple SAEs can be attached to the same model by calling `add_sae` multiple times.
* The hook name is automatically inferred from the SAE's metadata, but can be overridden with the `hook_name` keyword argument if needed.

### Steering a Feature

Once a SAE is attached, you can clamp a feature to a fixed value. This forces that feature's encoded activation to a constant for every token on every forward pass:
```python
sae.clamp(18493, 40.0)
```
Any call to `model.generate()` after this will have the clamp active. To remove all active interventions:
```python
sae.clear()
```

#### Notes:
* Multiple features can be clamped simultaneously by calling `clamp` multiple times.
* The clamp applies to every token position. For position-selective interventions, see [The Steering Dashboard](#the-steering-dashboard).

### Conditional Distributions

Instead of always fixing a feature to a value, you can apply it probabilistically. With probability `p`, the feature's activation is replaced with a sample from a distribution; otherwise it is left unchanged:
```python
from steering_utils import Dist

sae.cond_dist(18493, p=0.3, dist=Dist('normal', 40, 10))
```

#### Notes:
* `Dist` supports the following named distributions: `normal`, `uniform`, `lognormal`, `exponential`, `beta`. Parameters are passed positionally and correspond to the standard parameterization of each distribution.
* Clamps and conditional distributions can be active at the same time on different features.
* `model.cond_dist(feature_id, p, dist)` applies the same conditional distribution to all attached SAEs at once, if you have multiple.

### Generating Text

Generate text from the model with any active clamps or distributions applied:
```python
result = model.generate("I like sharks", max_tokens=100)
print(result)
```
This returns the full text (prompt + completion) as a string. By default, each generation is logged to `output/log.txt` as a JSON entry that includes the prompt, response, and all active SAE interventions.

#### Options:
* **max_tokens:** Integer, the number of new tokens to generate. Default is 100.
* **from_file:** If True, the prompt argument is treated as a file path and the prompt is read from that file. Default is False.
* **output_file:** If set to a file path string, the result will also be written to that file.
* **log:** If False, the generation will not be appended to `output/log.txt`. Default is True.

### Analyzing Features

You can run a prompt through the SAE and produce an interactive HTML analysis page that shows the top activated features per token:
```python
sae.analyze("I like sharks", html_output="output/analysis.html")
```
Open the resulting file in a browser. You will see a table of tokens with their top-N activated features, color-coded by activation strength. Clicking a feature shows its activation value at every token position. There is also a search bar to look up any feature by ID.

#### Options:
* **from_file:** If True, reads the prompt from a file. Default is False.
* **html_output:** File path to write the HTML to. If not set, the HTML string is returned but not saved.
* **top_n:** Number of top features to show per token. Default is 10.

### Interactive Chat

To chat interactively with the model with any active clamps applied:
```python
model.chat(max_tokens=200)
```
Type a prompt and press Enter to see the model's completion. Type `quit` to exit.

### Running Experiments

Experiments are defined in `experiment_scripts/` and registered in `experiments.py`. You can run any of them from the command line:
```
uv run experiments.py <experiment_name> [key value ...]
```
For example:
```
uv run experiments.py bulk_feature_stats
uv run experiments.py affectation_experiment mode 1
uv run experiments.py dashboard_experiment prompt "The ocean is" n_tokens 60
```
To see all available experiments:
```
uv run experiments.py
```

#### Notes:
* Key-value arguments after the experiment name are parsed automatically. Values are coerced to `int` first, then `float`, then kept as a string.
* Each experiment function has its own default arguments, so most can be run without providing any additional arguments.

### The Steering Dashboard

The steering dashboard is the primary experiment for this project. It lets you define many different steering strategies for a feature, run them all on a single prompt, and compare the results in an interactive browser-based dashboard.

To use it, import `SteeringExperiment` and the op classes you want, then register methods and run:
```python
from steering_utils import (
    load_gpt2_small, load_sae_from_neuronpedia, Dist,
    ClampOp, CondDistOp, EveryOtherTokenOp, FibonacciTokensOp, ZeroOp,
)
from experiment_scripts.steering_dashboard import SteeringExperiment

model = load_gpt2_small()
sae   = load_sae_from_neuronpedia("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
model.add_sae(sae)

exp = SteeringExperiment(model, feature_ids=[18493])
exp.add_method("baseline",    {})
exp.add_method("clamp_40",    {18493: ClampOp(40)})
exp.add_method("cond_dist",   {18493: CondDistOp(0.3, Dist("normal", 40, 10))})
exp.add_method("every_other", {18493: EveryOtherTokenOp(40)})
exp.add_method("zero_out",    {18493: ZeroOp()})

results = exp.run("I like sharks", n_tokens=100)
results.save_html("output/dashboard.html")
```
Open the output file in a browser. You will see:

* A **dropdown menu** at the top to select between the different steering methods you defined.
* A **token display** showing every token in the prompt and generated output. Tokens are colored in three ways:
  * **Gray:** The feature is inactive at this token.
  * **Green:** The feature is active and no intervention fired (the activation is natural).
  * **Orange:** The feature is active and an intervention changed its value.
* **Hovering over any token** shows a tooltip with the post-intervention and natural activation values for all tracked features. A ⚡ marker appears next to any feature where the intervention fired.
* A **feature activation chart** at the bottom shows a time series of the feature's value across all tokens, with a solid line for the post-intervention value and a dashed line for the natural value.

Saving calls `results.save_html()`, which writes the dashboard to both the path you specify and to a timestamped archive in `experiment_scripts/experiment_data/dashboard_experiment/`.

The built-in `dashboard_experiment` in `experiments.py` demonstrates all ten currently implemented op types and can be run directly with:
```
uv run experiments.py dashboard_experiment
```

#### Available Steering Ops:

All ops are importable from `steering_utils`. To define a custom op, subclass `SteeringOp` and implement `apply(current_val, token_idx)`.

| Op | Behavior |
|---|---|
| `ClampOp(value)` | Always fix the feature to a constant value |
| `CondDistOp(p, dist)` | With probability `p`, replace with a sample from `dist` |
| `EveryOtherTokenOp(value)` | Clamp on every other token (alternating) |
| `NthTokenOp(n, value, offset=0)` | Clamp every n-th token |
| `SpecificTokensOp(indices, value)` | Clamp only at the listed absolute token indices |
| `FibonacciTokensOp(value)` | Clamp on Fibonacci-indexed tokens (0, 1, 2, 3, 5, 8, 13, …) |
| `ThresholdOp(threshold, value)` | Clamp only when the natural activation exceeds the threshold |
| `ScaleOp(scale)` | Multiply the activation by a scale factor |
| `AddOp(delta)` | Add a constant to the activation |
| `ZeroOp()` | Always suppress the feature to zero |

#### Notes:
* A method registered with `{}` (empty ops dict) is a clean baseline with no intervention. All active tokens will appear green in the dashboard.
* Multiple feature IDs can be tracked simultaneously by passing a list to `feature_ids`. The dashboard will show values for all of them in the tooltip and will offer a dropdown to choose which feature drives the color coding.
* Token indices are absolute and include the BOS token at position 0.

## Project Structure

This section describes the internal architecture of the project. If you are simply using the interface, you do not need to understand every detail here. However, if you wish to modify the model, understand how interventions are applied, or add new steering operations, this section outlines the details.

### Dist

`Dist` is a simple wrapper around PyTorch distribution objects. It stores a named distribution with fixed parameters and exposes a `sample(shape)` method. It is used by `CondDistOp` and the older `cond_dist` API.

Supported names: `normal`, `uniform`, `lognormal`, `exponential`, `beta`.

### SteeringOp

`SteeringOp` is the abstract base class for all token-level steering operations. It defines one method:
```python
def apply(self, current_val: torch.Tensor, token_idx: int) -> torch.Tensor
```
`current_val` is a shape-(1,) tensor with the current encoded activation for one feature at one token. `token_idx` is the absolute position of that token in the sequence (including BOS). The method returns a modified tensor of the same shape.

Subclassing `SteeringOp` and implementing `apply` is all that is required to define a new operation. All built-in ops are listed in [Available Steering Ops](#available-steering-ops) above.

### SteerableSAE

`SteerableSAE` wraps a pretrained `sae_lens` SAE object. It is responsible for:
* Storing active clamps and conditional distributions (the older intervention API used by `generate` and `chat`)
* Running forward passes through the SAE to collect raw feature activations (`collect_activations`, `_forward_collect`)
* Generating the per-token analysis HTML page (`analyze`)
* Creating intervention hooks for use during generation (`_generation_hook`, `_make_tracking_hook`)

The key method for the dashboard experiment is `_make_tracking_hook(ops, track_fids)`. This creates a TransformerLens forward hook that:
1. Encodes the residual stream activations through the SAE
2. Snapshots the natural (pre-intervention) value for each tracked feature
3. Applies each `SteeringOp` at its corresponding feature index
4. Logs `{feature_id: {"v": post_op, "n": natural}}` for every token position
5. Decodes the modified activations back into the residual stream

The hook is KV-cache compatible: it tracks an absolute position counter that increments correctly whether the model processes one token at a time (with KV-cache) or the full growing sequence (without).

### SteerableModel

`SteerableModel` wraps a TransformerLens `HookedTransformer` and holds a list of attached `SteerableSAE` objects. It is the object a user interacts with for generation. Key methods:

* `add_sae(sae)` — Attaches a `SteerableSAE` to the model and registers the hook point.
* `generate(prompt, ...)` — Generates text with all active SAE clamps applied, logging to `output/log.txt`.
* `chat(max_tokens)` — Starts an interactive generation loop.
* `cond_dist(feature_id, p, dist)` — Applies a conditional distribution to a feature across all attached SAEs.

### Factory Functions

Three factory functions in `steering_utils.py` handle model and SAE loading:

* `load_gpt2_small()` — Loads GPT-2-small via TransformerLens. Returns a `SteerableModel`.
* `load_gemma_3_270m_it()` — Loads Gemma-3-270m-IT. Returns a `SteerableModel`.
* `load_sae_from_neuronpedia(release, sae_id, *, hook_name=None)` — Downloads a pretrained SAE from Neuronpedia using `sae_lens`. Returns a `SteerableSAE`. The hook name defaults to the one stored in the SAE's metadata.

### SteeringExperiment and ExperimentResults

These two classes live in `experiment_scripts/steering_dashboard.py` and form the backbone of the dashboard experiment.

`SteeringExperiment` orchestrates the experiment. The user registers named methods with `add_method(name, ops)`, then calls `run(prompt, n_tokens)`. For each method, it:
1. Creates a tracking hook from the method's ops via `sae._make_tracking_hook`
2. Adds the hook to the model
3. Calls `model.generate` with `use_past_kv_cache=True` to get output token IDs
4. Decodes the token IDs back to strings
5. Collects the per-token feature log from the hook closure
6. Resets all hooks and moves on to the next method

`ExperimentResults` stores all of the above data across every registered method and exposes `save_html(path)`, which renders the self-contained dashboard HTML and writes it to both the specified primary path and a timestamped archive copy.

The dashboard HTML is generated from a single template string with a JSON data blob injected at the placeholder `__DATA_JSON__`. All rendering logic is client-side JavaScript with no external dependencies.

### experiments.py

`experiments.py` is a thin dispatcher. It imports all experiment functions, stores them in an `EXPERIMENTS` dict, parses CLI key-value arguments, and calls the appropriate function. To add a new experiment, define a function in `experiment_scripts/`, import it here, and add it to `EXPERIMENTS`. That is all.

## Evaluation

This section is for grading purposes, as this repository is also an assignment (Hi Michael).

Per the evaluation criteria for this project, the following checklist aims to ensure all essential components are included:

1. The README. This file properly documents everything in the repository. Particularly, see [Repository Contents](#repository-contents), [Using the Project](#using-the-project), and [Project Structure](#project-structure).
2. The code does indeed run. Don't believe us? Try it yourself by running:
```
uv run experiments.py dashboard_experiment
```
3. Required components:
   * **Model and SAE Loading:** Implemented in the factory functions `load_gpt2_small()`, `load_gemma_3_270m_it()`, and `load_sae_from_neuronpedia()` in `steering_utils.py`.
   * **Feature Interventions:** All steering ops are implemented in `steering_utils.py`. The core intervention hook is `SteerableSAE._make_tracking_hook`.
   * **Multi-Method Comparison:** `SteeringExperiment` in `experiment_scripts/steering_dashboard.py` runs any number of intervention strategies on the same prompt.
   * **Interactive Dashboard:** `ExperimentResults.save_html()` produces a self-contained HTML file with token-level coloring, hover tooltips, and a feature activation chart.
   * **Extensibility:** New operations are added by subclassing `SteeringOp` and implementing `apply(current_val, token_idx)`.
4. Results and writeup are included in the [Results](#results) and [Design Choices, Challenges, Future Direction](#design-choices-challenges-future-direction) sections below.
5. Contributions are outlined in [Collaboration](#collaboration).

### Results

The following results describe observations from running the dashboard experiment with GPT-2-small, the `blocks.8.hook_resid_pre` SAE, and feature 18493 on the prompt "I like sharks".

Feature 18493 in this SAE is known to activate strongly on ocean and marine-animal related content. Under the `baseline` method (no intervention), this feature activates naturally at a moderate level on the word "sharks" and on generated tokens related to the ocean. The output is coherent and on-theme.

Under `clamp_40`, the feature is forced to an activation of 40 for every token. The model consistently produces more extreme ocean-related content, often generating lists of marine animals or descriptions of the deep sea that it would not produce at its natural activation level. Every generated token appears orange in the dashboard, since the intervention fires on every position.

Under `cond_dist_N(40,10)_p0.3`, roughly 30% of tokens receive an intervention drawn from N(40, 10). The remaining 70% are left to their natural activation. In the dashboard, these appear as a mix of orange (intervened), green (naturally active), and gray (inactive) tokens. The generated text tends to drift in and out of ocean-related content as the interventions fire stochastically.

Under `zero_out`, the feature is suppressed entirely. The model produces noticeably different text, often losing the thematic coherence around marine topics even though the prompt explicitly contains "sharks".

The most informative view is the feature activation chart in the dashboard, which shows the post-intervention line (solid) and natural line (dashed) together. For `ClampOp`, the post-intervention line is perfectly flat at 40 while the natural line varies normally. For `CondDistOp`, you can see individual spikes where the sampler fired. For `ZeroOp`, the post-intervention line is flat at zero while the natural line shows the true feature dynamics.

### Design Choices, Challenges, Future Direction

#### Challenges

One of the primary challenges in this project was **designing the intervention system to be both token-aware and easily extensible**. Early versions of the codebase applied clamps uniformly across all token positions using a simple dict. This worked for the basic use case but made it impossible to ask questions like "what happens if I only steer every other token?" or "what if I only intervene when the feature is already active?" The solution was to abstract interventions into `SteeringOp` objects with a per-token `apply(current_val, token_idx)` interface, which cleanly separates the logic of each intervention type from the mechanics of applying it.

A significant technical challenge was **tracking feature values during generation in a way that is compatible with KV-cache**. Without KV-cache, TransformerLens processes the growing context sequence on every forward pass, so the same token positions are seen by the hook many times. With KV-cache, the hook receives only the new token on each generation step. The solution was an absolute position counter inside the hook closure that increments by the `seq_len` of each hook call, regardless of whether that is 1 (KV-cache) or many (no cache). This ensures each absolute token position is logged exactly once, at the time it is first processed.

Another challenge was **representing "did the intervention fire?" at each token** for the three-color dashboard display. Storing only the post-intervention value makes it impossible to tell whether the op changed anything. The solution was to snapshot natural values before applying ops and store both `{"v": post_op, "n": natural}` for every tracked feature at every position. Comparing these two values with a small epsilon threshold reliably identifies which tokens were actually intervened on.

#### Design Choices

The most significant design choice was to make **the SAE hook the single intervention point for both generation and tracking**. Rather than having separate code paths for "generate with steering" and "analyze activations", the same `_make_tracking_hook` method is used for both. This means that the values shown in the dashboard are exactly the values that passed through the SAE during generation, with no separate analysis pass required afterward.

The decision to keep **`SteeringOp` subclasses stateless with respect to the sequence** was deliberate. Each op sees only `(current_val, token_idx)` and nothing about the history of prior activations. This makes ops trivially composable and testable, at the cost of not being able to express things like "clamp this token if the previous token was also clamped." Stateful ops would require a more complex interface and are left for future work.

The **HTML dashboard as a self-contained file** was chosen because it makes outputs easy to share, archive, and open without a running server. All data is embedded as a JSON blob and all rendering is done in plain JavaScript. This comes with the tradeoff that very long generations with many tracked features produce large files, but this has not been a practical problem at the scales we are working with.

#### Future Work

Future work will primarily involve three things:
1. **More intervention types.** The current op library covers the cases we have needed so far, but there are many more interesting operations — for example, ops that condition on the activations of other features, ops that learn intervention schedules from data, or ops that implement feedback control toward a target activation level.
2. **Multi-SAE experiments.** The current dashboard supports multiple SAEs being attached to a model, but `SteeringExperiment` applies ops to only one SAE at a time. Extending the experiment interface to support specifying different ops per-SAE would allow for studying how interventions at different layers interact.
3. **Automated feature discovery.** Right now, the user must already know which feature ID they want to steer. Integrating the `bulk_feature_stats` and `affectations` experiments more tightly with the dashboard would allow the user to discover interesting features automatically and then study them through the dashboard in a single workflow.

### Collaboration

This project was completed by Andy Holmberg and Stone Amsbaugh. Both team members contributed significantly to the final product and hold a deep understanding of how it works.

Andy was primarily responsible for the research direction and experiment design. He led the investigation into which features to study, determined the experimental methodology for the affectations experiments and the hot/cold word substitution approach, and contributed to the analysis of results.

Stone was primarily responsible for the engineering of the project. He designed and implemented the `SteerableSAE` and `SteerableModel` classes, the intervention hook system, the `SteeringOp` hierarchy, the `SteeringExperiment` runner, and the HTML dashboard. He also maintained the project structure, the experiment runner CLI, and this documentation.

Generative AI was used for exactly one part of this project: the HTML dashboard template. The interactive token display, tooltip design, SVG chart, and color logic in the dashboard were developed with AI assistance. All Python logic defining how interventions are applied and how data is collected was written by the authors.

## Attribution

This project loads pretrained SAEs from [Neuronpedia](https://www.neuronpedia.org/) using the [sae-lens](https://github.com/jbloomAus/SAELens) library. All SAEs used in this project belong to their respective creators and are accessed through their public APIs.

Model weights are loaded via [TransformerLens](https://github.com/neelnanda-io/TransformerLens), which provides the hook-based forward pass infrastructure that makes token-level interventions possible.

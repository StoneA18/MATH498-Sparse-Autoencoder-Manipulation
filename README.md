# Identifying Intervention Methods on Sparse Autoencoders for Influencing Model Behavior 
### Authors: Andy Holmberg and Stone Amsbaugh
#### MATH498 - Decoding GPT. Spring 2026. Taught by Dr. Michael Ivanitsky

## Welcome to the Repository
If you are wondering what this project is about (or if you are grading a project update), then the best place to start is with our most recent project update. 

[Most recent update](./docs/update_042226.pdf)

This, and similar required documents for the coursework-side of this project, can all be found within the `/docs` directory. Specifically, the required questions and contributions documentation for this update can be found here: [Questions and Contributions 04/22/26](/docs/update_042226_questions_contributions.md)

As for using this code, you are welcome to play around with it, but it is admittedly not in a final, deliverable state. Some guidance on how to interact with it is included below.

## Repository Guide

#### Setup
We recommend using `uv` for this repository. See: [uv package manager](https://docs.astral.sh/uv/getting-started/installation/). 
Once `uv` is installed, run `uv sync` to collect the required packages.

#### Feature Steering
All logic pertaining to loading models, SAEs, exploring features, and intervening is contained within `feature_steering.py`. This file is a utility file, and it is meant to be used as an API for performing useful operations. Feel free to check it out.

#### Exploration
An interface for interacting with the feature steering utilities is provided in `explore.py`. Details for using this are included below.

#### Docs
This project isn't just really cool, it is also for a course. Course-specific documents and deliverables are located in the `/docs` directory.

## Exploring the Project
If you have already read our most recent writeup in `\docs` and are looking to experiment with some of these interfaces, you can open the CLI by executing:
`uv run explore.py`
When first run, this script will download the required models. You may need a hugging face key to download some models. As this project is still in progress, documentation for this step is not yet provided, and this isn't necessary for all models.

To ensure you have a model you can query, try running:
`generate -f samples/sample3.txt`
This will query your model using the prompt in the referenced file. You can limit the output tokens with the `-n` flag as well.

Now, try clamping a feature. Do this with:
`clamp 18493 40`
This fixes feature 18493 at a value of 40 for all tokens. Try generating from the model again, and observe the differences. You can reset the clamps with `clear`. 

Lastly, try:
`explore -f samples/sample3.txt`
When this is done, open `explorationoutput.html` in a browser, and you should see a nice dashboard for exploring the features that were activated for the tokens in this prompt.

Now you have a hang of the simple premise of this interface. There are lots of other commands for intervening and more being added, and you can always get help with these by typing `help` to see all commands, or `help {command}` for command specific information.



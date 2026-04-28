# SAE Projection Playground Deployment

The playground is a Python web app served by `web/web_sae_playground.py`. It loads the
TransformerLens model and local `TrainableSAE` checkpoint lazily on the first
analysis or generation request.

## Local

```bash
uv run python web/web_sae_playground.py --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

Useful flags:

```bash
--device cuda
--model-dtype bfloat16
--local-files-only
--sae-root custom_saes
--sae-root saved_saes
```

## Google Cloud

For quick demos, a GPU VM is the least surprising option: clone the repo, install
`uv`, run `uv sync`, make sure Hugging Face auth/model files are available, then
serve with:

```bash
uv run python web/web_sae_playground.py --host 0.0.0.0 --port 8000 --device cuda --model-dtype bfloat16
```

Cloud Run can work for CPU demos, but model download and cold starts can be
annoying. If using Cloud Run, bake dependencies into the image and either bake in
the model cache/checkpoints or mount/cache them so each cold start does not
download the model again.

## Other Good Hosts

Hugging Face Spaces is a natural fit for a public class/demo version because it
already understands model artifacts and secrets. A small GPU VM on RunPod, Lambda,
or GCP is better if you want responsive generation while experimenting.

Keep the app private if you expose unrestricted generation. The current server has
no authentication or rate limiting.

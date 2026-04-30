import csv
import sys
from pathlib import Path

from steering_utils import load_gpt2_small, load_sae_from_neuronpedia

def bulk_feature_stats(text_path=None):
    """
    Pass a block of text through GPT-2-small's layer-8 residual-stream SAE,
    collect per-feature activation statistics across all tokens, and write
    feature_stats.csv with: feature_id, count, mean_when_active, std_when_active.
    """
    model = load_gpt2_small()
    sae = load_sae_from_neuronpedia("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    model.add_sae(sae)

    if text_path is None:
        text_path = Path('experiment_scripts/experiment_data/bulk_features_sample_text.txt')

    print("Collecting activations …")
    feature_matrix = sae.collect_activations(str(text_path), from_file=True)
    # shape: (n_tokens, n_features)
    n_tokens, n_features = feature_matrix.shape
    print(f"  {n_tokens} tokens  |  {n_features} features")

    # --- compute per-feature stats (only where active) ---
    rows: list[tuple] = []
    for fid in range(n_features):
        vals = feature_matrix[:, fid]
        active = vals[vals != 0]
        if len(active) == 0:
            continue
        count = int(len(active))
        mean  = float(active.mean())
        std   = float(active.std()) if count > 1 else 0.0
        rows.append((fid, count, mean, std))

    rows.sort(key=lambda r: -r[1])  # most-frequently-active first

    # --- write CSV ---
    out_path = Path("feature_stats.csv")
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["feature_id", "count", "mean_when_active", "std_when_active"])
        writer.writerows(rows)

    print(f"Wrote {out_path}  ({len(rows)} active features)")
    print("\nTop 10 most frequent features:")
    print(f"  {'feature_id':>10}  {'count':>6}  {'mean':>8}  {'std':>8}")
    for fid, count, mean, std in rows[:10]:
        print(f"  {fid:>10}  {count:>6}  {mean:>8.4f}  {std:>8.4f}")


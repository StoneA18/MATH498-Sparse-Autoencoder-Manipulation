import csv
from pathlib import Path

import torch

from steering_utils import load_gpt2_small, load_sae_from_neuronpedia

_STATS_PATH = Path("experiment_scripts/experiment_data/affectations/affectation_stats.csv")


def affectation_experiment(mode=None):
    """
    Compare SAE feature activations between happy and sad text corpora.

    mode=None / 0  — collect activations from both corpora, compute per-feature
                     stats, and write affectation_stats.csv.

    mode=1         — load the existing affectation_stats.csv, clamp the top 10
                     most positively-biased features (highest happy - sad count)
                     to their mean_happy activation value, then start an
                     interactive chat session with the steered model.

    CSV columns:
      feature_id,
      count_total, count_happy, count_sad, count_diff (happy - sad),
      mean_total,  mean_happy,  mean_sad,  mean_diff  (happy - sad)
    """
    model = load_gpt2_small()
    sae = load_sae_from_neuronpedia("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    model.add_sae(sae)

    # ------------------------------------------------------------------
    # mode 1: load existing stats, clamp, chat
    # ------------------------------------------------------------------
    if mode == 1:
        print("Clamping most biased features...")
        if not _STATS_PATH.exists():
            print(f"  {_STATS_PATH} not found — run without mode=1 first.")
            return

        positive_rows: list[tuple[int, int, float]] = []
        with _STATS_PATH.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                diff = int(row["count_diff"])
                if diff > 0:
                    positive_rows.append((
                        int(row["feature_id"]),
                        diff,
                        float(row["mean_happy"]),
                    ))

        positive_rows.sort(key=lambda r: -r[1])  # highest count_diff first
        top10 = positive_rows[:10]

        print(f"  Clamping {len(top10)} features to their mean_happy activation:")
        print(f"  {'feature':>8}  {'count_diff':>10}  {'mean_happy':>10}")
        for fid, diff, mean_h in top10:
            print(f"  {fid:>8}  {diff:>+10}  {mean_h:>10.4f}")
            sae.clamp(fid, mean_h)

        print()
        model.chat()
        return

    # ------------------------------------------------------------------
    # mode 0 / None: full analysis → write CSV
    # ------------------------------------------------------------------
    happy_text_path = Path("experiment_scripts/experiment_data/affectations/very_happy.txt")
    sad_text_path   = Path("experiment_scripts/experiment_data/affectations/very_sad.txt")

    print("Collecting activations for happy text...")
    happy_mat = sae.collect_activations(str(happy_text_path), from_file=True)
    print(f"  {happy_mat.shape[0]} tokens  |  {happy_mat.shape[1]} features")

    print("Collecting activations for sad text...")
    sad_mat = sae.collect_activations(str(sad_text_path), from_file=True)
    print(f"  {sad_mat.shape[0]} tokens  |  {sad_mat.shape[1]} features")

    n_features = min(happy_mat.shape[1], sad_mat.shape[1])

    rows: list[tuple] = []
    for fid in range(n_features):
        h_active = happy_mat[:, fid]
        h_active = h_active[h_active != 0]
        s_active = sad_mat[:, fid]
        s_active = s_active[s_active != 0]

        count_happy = int(len(h_active))
        count_sad   = int(len(s_active))
        count_total = count_happy + count_sad

        if count_total == 0:
            continue

        mean_happy = float(h_active.mean()) if count_happy > 0 else 0.0
        mean_sad   = float(s_active.mean()) if count_sad   > 0 else 0.0
        mean_total = float(torch.cat([h_active, s_active]).mean()) \
                     if (count_happy > 0 and count_sad > 0) \
                     else (mean_happy or mean_sad)

        rows.append((
            fid,
            count_total, count_happy, count_sad, count_happy - count_sad,
            round(mean_total, 6), round(mean_happy, 6), round(mean_sad, 6),
            round(mean_happy - mean_sad, 6),
        ))

    rows.sort(key=lambda r: -abs(r[4]))  # most biased first

    _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _STATS_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "feature_id",
            "count_total", "count_happy", "count_sad", "count_diff",
            "mean_total",  "mean_happy",  "mean_sad",  "mean_diff",
        ])
        writer.writerows(rows)

    print(f"\nWrote {_STATS_PATH}  ({len(rows)} active features)")
    print("\nTop 10 most differentially-activated features (by |count_happy - count_sad|):")
    print(f"  {'fid':>6}  {'tot':>5}  {'hap':>5}  {'sad':>5}  {'Δcnt':>6}  "
          f"{'mean_h':>8}  {'mean_s':>8}  {'Δmean':>8}")
    for row in rows[:10]:
        fid, ct, ch, cs, cd, _, mh, ms, md = row
        print(f"  {fid:>6}  {ct:>5}  {ch:>5}  {cs:>5}  {cd:>+6}  "
              f"  {mh:>8.4f}  {ms:>8.4f}  {md:>+8.4f}")

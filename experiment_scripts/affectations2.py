import csv
import datetime
import math
from pathlib import Path
import random
import torch

from steering_utils import load_gpt2_small, load_sae_from_neuronpedia, Dist

_STATS_PATH  = Path("experiment_scripts/experiment_data/affectations2/affectations2_stats.csv")
_LOG_PATH    = Path("experiment_scripts/experiment_data/affectations2/responses.log")
_CLAMP_PROMPT = "The stove is"
_OUT_TOKENS = 150

HOT_WORDS  = ['hot','warm','scorching','blazing','sweltering','sizzling','boiling','torrid','searing','broiling']
COLD_WORDS = ['cold','chilly','freezing','frigid','frosty','glacial','wintry','icy','nippy','arctic']


def _log(text: str) -> None:
    """Append text to responses.log with a timestamp header."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[{ts}]\n{text}\n")


def affectation_experiment_v2(mode=None, top_n=10):
    """
    mode=None/0  — collect activations from fib_sentences, compute per-feature
                   stats, write affectations2_stats.csv.

    mode=1       — load existing stats, clamp top_n hot-biased features
                   (positive delta_total_times_active only) to their
                   average_activated_value_hot, then generate a response to
                   _CLAMP_PROMPT with and without clamps.  Both responses are
                   printed and appended to responses.log.
    """
    model = load_gpt2_small()
    sae   = load_sae_from_neuronpedia("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    model.add_sae(sae)

    # ------------------------------------------------------------------
    # mode 1: load stats, clamp top-N hot-biased features, compare output
    # ------------------------------------------------------------------
    if mode == 1:
        if not _STATS_PATH.exists():
            print(f"  {_STATS_PATH} not found — run without mode=1 first.")
            return

        candidates: list[tuple[int, int, float, float]] = []
        with _STATS_PATH.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                delta = int(row["delta_total_times_active"])
                if delta > 0:
                    candidates.append((
                        int(row["feature_id"]),
                        delta,
                        float(row["average_activated_value_hot"]),
                        float(row['std_activated_value_hot']),
                    ))

        candidates.sort(key=lambda r: -r[1])
        top = candidates[:top_n]

        # baseline — no clamps
        baseline = model.generate(_CLAMP_PROMPT, max_tokens=_OUT_TOKENS)
        print(f"\n--- Baseline (no clamps) ---\n{baseline}\n")

        # apply clamps
        print(f"Clamping top {len(top)} hot-biased features:")
        print(f"  {'feature':>8}  {'delta_count':>11}  {'avg_hot':>9}")
        for fid, delta, avg_hot, std in top:
            print(f"  {fid:>8}  {delta:>+11}  {avg_hot:>9.4f}")
            sae.clamp(fid, 1.5*avg_hot)

        steered = model.generate(_CLAMP_PROMPT, max_tokens=_OUT_TOKENS)
        sae.clear()
        print(f"\n--- Steered (top {len(top)} features clamped to avg_hot) ---\n{steered}\n")

         # apply cond_dist
        sae.clear()
        print(f"Conditional;y distributing top {len(top)} hot-biased features:")
        print(f"  {'feature':>8}  {'delta_count':>11}  {'avg_hot':>9}")
        for fid, delta, avg_hot, std in top:
            print(f"  {fid:>8}  {delta:>+11}  {avg_hot:>9.4f}")
            dist = Dist('normal', avg_hot*1.5, std)
            sae.cond_dist(fid, 0.3, dist)

        steered2 = model.generate(_CLAMP_PROMPT, max_tokens=_OUT_TOKENS)
        sae.clear()
        print(f"\n--- Steered (top {len(top)} features conditionally distributed to avg_hot) ---\n{steered2}\n")

        entry = (
            f"prompt: {_CLAMP_PROMPT!r}  |  top_n={len(top)}\n"
            f"\n[baseline]\n{baseline}\n"
            f"\n[steered — {len(top)} features clamped to avg_hot]\n{steered}\n"
            f"\n[steered2 — {len(top)} features conditionally distributed to avg_hot]\n{steered2}\n"
            f"\nclamped features: {[(fid, round(avg_hot, 4)) for fid, _, avg_hot, _ in top]}\n"
        )
        _log(entry)
        print(f"Appended to {_LOG_PATH}")
        return

    fib_path = Path("experiment_scripts/experiment_data/affectations2/fib_sentences.txt")

    # feature_id -> {'hot': [activation, ...], 'cold': [activation, ...]}
    feature_vals: dict[int, dict[str, list[float]]] = {}

    lines = fib_path.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        # strip leading "N. " numbering
        body = " ".join(line.split()[1:])
        blank_idx = body.index("<blank>")
        prefix, suffix = body[:blank_idx], body[blank_idx + len("<blank>"):]

        hot_word  = random.choice(HOT_WORDS)
        cold_word = random.choice(COLD_WORDS)
        hot_sentence  = prefix + hot_word  + suffix
        cold_sentence = prefix + cold_word + suffix

        print(f"[{i+1}/{total}] hot:  {hot_sentence}")
        print(f"[{i+1}/{total}] cold: {cold_sentence}")

        hot_mat  = sae.collect_activations(hot_sentence,  from_file=False)
        cold_mat = sae.collect_activations(cold_sentence, from_file=False)

        n_features = min(hot_mat.shape[1], cold_mat.shape[1])

        for fid in range(n_features):
            h_vals = hot_mat[:, fid]
            c_vals = cold_mat[:, fid]

            h_active = h_vals[h_vals != 0].tolist()
            c_active = c_vals[c_vals != 0].tolist()

            if not h_active and not c_active:
                continue

            if fid not in feature_vals:
                feature_vals[fid] = {'hot': [], 'cold': []}
            feature_vals[fid]['hot'].extend(h_active)
            feature_vals[fid]['cold'].extend(c_active)

    # ------------------------------------------------------------------
    # Compute per-feature stats
    # ------------------------------------------------------------------
    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _std(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        m = _mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    rows: list[tuple] = []
    for fid, buckets in feature_vals.items():
        h = buckets['hot']
        c = buckets['cold']
        combined = h + c

        total_times_active      = len(combined)
        total_times_active_hot  = len(h)
        total_times_active_cold = len(c)

        avg_total = _mean(combined)
        avg_hot   = _mean(h)
        avg_cold  = _mean(c)

        std_total = _std(combined)
        std_hot   = _std(h)
        std_cold  = _std(c)

        delta_count = total_times_active_hot - total_times_active_cold
        delta_avg   = avg_hot - avg_cold

        rows.append((
            fid,
            total_times_active,      round(avg_total, 6), round(std_total, 6),
            total_times_active_hot,  round(avg_hot,   6), round(std_hot,   6),
            total_times_active_cold, round(avg_cold,  6), round(std_cold,  6),
            delta_count, round(delta_avg, 6),
        ))

    rows.sort(key=lambda r: -abs(r[10]))  # sort by |delta_total_times_active|

    _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _STATS_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "feature_id",
            "total_times_active",      "average_activated_value",      "std_activated_value",
            "total_times_active_hot",  "average_activated_value_hot",  "std_activated_value_hot",
            "total_times_active_cold", "average_activated_value_cold", "std_activated_value_cold",
            "delta_total_times_active", "delta_average_activated_value",
        ])
        writer.writerows(rows)

    print(f"\nWrote {_STATS_PATH}  ({len(rows)} active features)")
    print(f"\nTop 10 by |delta_total_times_active|:")
    print(f"  {'fid':>6}  {'tot':>5}  {'hot':>5}  {'cold':>5}  {'Δcnt':>6}"
          f"  {'avg_h':>8}  {'avg_c':>8}  {'Δavg':>8}")
    for row in rows[:10]:
        fid, tot, avg_t, std_t, tot_h, avg_h, std_h, tot_c, avg_c, std_c, dcnt, davg = row
        print(f"  {fid:>6}  {tot:>5}  {tot_h:>5}  {tot_c:>5}  {dcnt:>+6}"
              f"  {avg_h:>8.4f}  {avg_c:>8.4f}  {davg:>+8.4f}")

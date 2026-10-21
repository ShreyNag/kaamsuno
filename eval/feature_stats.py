"""Per-feature vote rates on speech vs non-speech frames, per condition.
A useful feature has a high rate on speech and a low rate on noise.

    python -m eval.feature_stats   -> eval/out/feature_vote_rates.md
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from app.audio_io import dev_dirs, load_labels, load_wav, segments_to_frames
from app.config import EVAL, ROOT
from app.vad.features import compute_features, feature_votes
from eval.mix_noise import condition_rng, mix_at_snr


def main() -> None:
    clips_dir, noise_dir = dev_dirs()
    labels = load_labels(clips_dir / "vad_labels.csv")
    noises = {n: load_wav(noise_dir / f"{n}.wav") for n in EVAL.noise_types}
    conds = [("clean", None, None)] + [(f"{n} {s:g}dB", n, s) for n in noises for s in EVAL.snrs_db]

    # cond -> feature -> [speech hits, speech total, noise hits, noise total]
    acc: dict[str, dict[str, np.ndarray]] = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    for clip, segs in labels.items():
        x = load_wav(clips_dir / clip)
        for name, n, snr in conds:
            y = x if n is None else mix_at_snr(x, noises[n], snr, condition_rng(clip, n, snr), segs)
            f = compute_features(y)
            truth = segments_to_frames(segs, f["t"])
            votes = {"A gate": f["gate"], **{k: v for k, v in feature_votes(f).items()}}
            for k, v in votes.items():
                acc[name][k] += [(v & truth).sum(), truth.sum(), (v & ~truth).sum(), (~truth).sum()]

    feats = list(next(iter(acc.values())))
    lines = ["# Per-feature vote rate: speech frames / non-speech frames (higher / lower is better)", "",
             "| condition | " + " | ".join(feats) + " |", "|---" * (len(feats) + 1) + "|"]
    for name in acc:
        cells = [f"{a[0] / a[1]:.2f} / {a[2] / a[3]:.2f}" for a in (acc[name][k] for k in feats)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    out = ROOT / "eval" / "out" / "feature_vote_rates.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""eval/vad_eval.py and eval/auto_label.py: the parts that decide the numbers."""

from __future__ import annotations

import numpy as np

from app.audio_io import load_labels, load_wav, segments_to_frames
from app.config import DATA_DIR
from eval.auto_label import label_clip
from eval.vad_eval import counts, metrics, pipecat_counter, to_frames


def test_pipecat_counter_matches_the_analyzer_state_machine():
    # 20 ms chunks, start/stop 0.2 s = 10 chunks
    dec = np.array([0] * 5 + [1] * 30 + [0] * 3 + [1] * 10 + [0] * 20, dtype=bool)
    out = pipecat_counter(dec, 0.02, 0.2, 0.2)
    assert not out[:14].any() and out[14]  # SPEAKING after 10 speech chunks
    assert out[35:38].all()  # a 3-chunk dip stays in STOPPING (still speech)
    first_quiet = 48 + 9  # 10th non-speech chunk after the last speech
    assert out[first_quiet - 1] and not out[first_quiet:].any()


def test_metrics_weight_recall():
    m = metrics(np.array([80, 10, 20, 90]))  # tp fp fn tn
    assert round(m["precision"], 3) == 0.889 and m["recall"] == 0.8
    assert m["f2"] < m["f1"] < m["precision"]  # F2 leans towards the (lower) recall
    assert m["false_alarm"] == 0.1


def test_to_frames_and_counts():
    truth = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=bool)  # 10 ms frames
    pred = to_frames(np.array([0, 1, 1, 0], dtype=bool), 0.02, 8)  # 20 ms chunks
    assert pred.tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert counts(pred, truth).tolist() == [4, 0, 0, 4]


def test_auto_labels_cover_all_speech_of_the_synthetic_clips():
    lab = load_labels(DATA_DIR / "vad_dev_synth" / "vad_labels.csv")
    for name in sorted(lab)[:3]:
        x = load_wav(DATA_DIR / "vad_dev_synth" / name)
        auto, _ = label_clip(x)
        t = np.arange(0, len(x) / 16000, 0.01) + 0.005
        y, a = segments_to_frames(lab[name], t), segments_to_frames(auto, t)
        assert a[y].mean() > 0.99  # generous: never miss speech
        assert a[~y].mean() < 0.45  # but don't swallow the pauses


def test_offline_replay_equals_the_live_pipecat_analyzers():
    """The 'deployed' view must be exactly what the bot's VADProcessor would see."""
    import asyncio

    from pipecat.audio.vad.vad_analyzer import VADState

    from app.turns import build_vad
    from eval.mix_noise import condition_rng, mix_at_snr
    from eval.vad_eval import CHUNK_S, confidences

    x = load_wav(DATA_DIR / "vad_dev_synth" / "clip_00.wav")
    x = mix_at_snr(x, load_wav(DATA_DIR / "noise_synth" / "drilling.wav"), 5, condition_rng("clip_00", "d", 5))
    xq = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    cc = confidences(xq.astype(np.float32) / 32768)
    n = min(len(cc.silero), len(cc.custom_share32))
    for mode, chunk, dec in (
        ("custom", 320, cc.custom_share20 >= 0.5),
        ("composite", 512, (cc.silero[:n] >= 0.7) & (cc.custom_share32[:n] >= 0.5)),
    ):
        vad = build_vad(mode)
        vad.set_sample_rate(16000)
        pcm = xq.tobytes()

        async def live():
            return [await vad.analyze_audio(pcm[o : o + chunk * 2]) in (VADState.SPEAKING, VADState.STOPPING)
                    for o in range(0, len(pcm) - chunk * 2 + 1, chunk * 2)]

        real, sim = np.array(asyncio.run(live())), pipecat_counter(dec, CHUNK_S[mode])
        k = min(len(real), len(sim))
        assert (real[:k] == sim[:k]).all(), mode

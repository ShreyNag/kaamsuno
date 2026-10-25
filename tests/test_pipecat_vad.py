"""The custom / composite VAD as Pipecat VADAnalyzers, driven offline through Pipecat's own
`analyze_audio` (start/stop counters included), on the synthetic dev clips and the turn fixture."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from pipecat.audio.vad.vad_analyzer import VADState

from app.audio_io import load_labels, load_wav
from app.config import DATA_DIR, SAMPLE_RATE, VAD
from app.turns import build_vad
from app.vad import custom_vad
from app.vad.pipecat_vad import CompositeVADAnalyzer, CustomVADAnalyzer

FIX = Path(__file__).parent / "fixtures"
SYNTH = DATA_DIR / "vad_dev_synth"
CHUNK = 320  # 20 ms, like WebRTC input


def to_pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


def speaking_track(vad, x: np.ndarray) -> list[tuple[float, bool]]:
    """(time at the end of each 20 ms chunk, user is speaking) as Pipecat's VADProcessor sees it:
    speaking from SPEAKING until the state reaches QUIET (STOPPING is still speaking)."""
    vad.set_sample_rate(SAMPLE_RATE)
    pcm = to_pcm(x)

    async def run():
        out = []
        for i, off in enumerate(range(0, len(pcm), CHUNK * 2)):
            state = await vad.analyze_audio(pcm[off : off + CHUNK * 2])
            out.append(((i + 1) * CHUNK / SAMPLE_RATE, state in (VADState.SPEAKING, VADState.STOPPING)))
        return out

    return asyncio.run(run())


def stop_times(track: list[tuple[float, bool]]) -> list[float]:
    return [t for (_, a), (t, b) in zip(track, track[1:]) if a and not b]


def test_build_vad_modes():
    assert type(build_vad("custom")) is CustomVADAnalyzer
    assert type(build_vad("composite")) is CompositeVADAnalyzer
    with pytest.raises(ValueError):
        build_vad("magic")


def test_custom_confidence_is_the_offline_candidate_share():
    """Streaming through the Pipecat wrapper gives the same per-frame candidates as process_file."""
    x = load_wav(SYNTH / "clip_00.wav")
    _, offline = custom_vad.process_file(x)
    vad = CustomVADAnalyzer()
    vad.set_sample_rate(SAMPLE_RATE)
    pcm = to_pcm(x)
    stream = vad._stream.vad
    got = []
    for off in range(0, len(pcm), CHUNK * 2):
        got += [r.candidate for r in stream.push(pcm[off : off + CHUNK * 2])]
    cand = offline["candidate"][: len(got)]
    # int16 quantisation can flip a frame at the gate threshold; nothing more
    assert np.mean(np.array(got) == cand) > 0.995


@pytest.mark.parametrize("mode", ["custom", "composite"])
def test_live_vad_finds_the_labelled_speech(mode):
    labels = load_labels(SYNTH / "vad_labels.csv")
    hits, total, false_t, quiet_t = 0, 0, 0.0, 0.0
    for name in sorted(labels)[:4]:
        track = speaking_track(build_vad(mode), load_wav(SYNTH / name))
        segs = labels[name]
        for t, sp in track:
            in_speech = any(s <= t <= e for s, e in segs)
            # skip the start/stop transition zones (start_secs / stop_secs) around each label edge
            near_edge = any(abs(t - s) < 0.3 or abs(t - e) < 0.3 for s, e in segs)
            if near_edge:
                continue
            if in_speech:
                total += 1
                hits += sp
            else:
                quiet_t += 1
                false_t += sp
    recall, false_alarm = hits / total, false_t / quiet_t
    print(f"{mode}: recall {recall:.3f}, false alarm {false_alarm:.3f}")
    assert recall > 0.9
    assert false_alarm < 0.1


def test_custom_vad_stops_in_the_mid_sentence_pause():
    meta = json.loads((FIX / "mid_sentence_pause.json").read_text())
    track = speaking_track(build_vad("custom"), load_wav(FIX / "mid_sentence_pause.wav"))
    stops = stop_times(track)
    pause_start, _ = meta["pause"]
    speech_end = meta["speech"][-1][1]
    # exactly one stop per phrase, VAD_STOP_SECS (0.2 s) after its end, within a few frames
    assert len(stops) == 2, stops
    assert stops[0] == pytest.approx(pause_start + 0.2, abs=0.08)
    assert stops[1] == pytest.approx(speech_end + 0.2, abs=0.08)


def test_composite_is_silero_gated_by_custom():
    x = load_wav(SYNTH / "clip_01.wav")
    pcm = to_pcm(x)
    vad = CompositeVADAnalyzer()
    vad.set_sample_rate(SAMPLE_RATE)
    n = vad.num_frames_required() * 2
    for off in range(0, len(pcm) - n, n):
        conf = vad.voice_confidence(pcm[off : off + n])
        if vad.last_custom_share < 0.5:
            assert conf == 0.0


def test_composite_rejects_white_noise_that_silero_might_take():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, SAMPLE_RATE * 5).astype(np.float32)
    track = speaking_track(build_vad("composite"), noise)
    assert not any(sp for _, sp in track)
    assert VAD.sample_rate == SAMPLE_RATE

#!/usr/bin/env python3
"""Speaker diarization via sherpa-onnx (no HF gate).

Usage: diarize.py <wav_path> [num_speakers]
Prints JSON: [{"start": float, "end": float, "speaker": int}, ...]
num_speakers <= 0 (or omitted) -> auto-detect via clustering threshold.
"""
import json
import os
import sys
import wave

import numpy as np
import sherpa_onnx

HERE = "/home/pbrown/s2t/diar-models"
SEG = f"{HERE}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB = os.environ.get("DIAR_EMB", f"{HERE}/emb.onnx")
THRESHOLD = float(os.environ.get("DIAR_THRESHOLD", "0.5"))


def read_wav_mono_f32(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, "expected 16-bit PCM"
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def build(num_speakers):
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers and num_speakers > 0 else -1,
            threshold=THRESHOLD,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not cfg.validate():
        print("invalid diarization config", file=sys.stderr)
        sys.exit(2)
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)


def main():
    path = sys.argv[1]
    num = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    audio, sr = read_wav_mono_f32(path)
    sd = build(num)
    if sr != sd.sample_rate:
        # sherpa expects the model's rate (16k); resample by linear interp.
        import math

        tgt = sd.sample_rate
        idx = np.linspace(0, len(audio) - 1, int(math.ceil(len(audio) * tgt / sr)))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    result = sd.process(audio).sort_by_start_time()
    out = [{"start": r.start, "end": r.end, "speaker": r.speaker} for r in result]
    print(json.dumps(out))


if __name__ == "__main__":
    main()

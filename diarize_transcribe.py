#!/usr/bin/env python3
"""Diarized transcription: WAV -> speaker-labeled transcript.

Combines faster-whisper (words + timestamps) with sherpa-onnx speaker
diarization (who spoke when), assigning each transcript segment to the speaker
whose diarization turn overlaps it most.

Usage: diarize_transcribe.py <wav_path> [num_speakers]
Prints JSON: {"segments": [{"speaker": int, "start": float, "end": float,
"text": str}], "text": "[S0] ... [S1] ..."}
"""
import json
import os
import sys
import wave

import numpy as np
import sherpa_onnx
from faster_whisper import WhisperModel

HERE = "/home/pbrown/s2t/diar-models"
SEG = f"{HERE}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB = os.environ.get("DIAR_EMB", f"{HERE}/titanet.onnx")
WHISPER_MODEL = os.environ.get("DIAR_WHISPER", "tiny")
THRESHOLD = float(os.environ.get("DIAR_THRESHOLD", "0.5"))


def read_wav_mono_f32(path):
    with wave.open(path, "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def diarize(audio, num_speakers):
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
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    turns = sd.process(audio).sort_by_start_time()
    return [(t.start, t.end, t.speaker) for t in turns]


def speaker_for(start, end, turns):
    """The speaker whose diarization turn overlaps [start,end] the most."""
    best, best_ov = None, 0.0
    for ts, te, spk in turns:
        ov = max(0.0, min(end, te) - max(start, ts))
        if ov > best_ov:
            best, best_ov = spk, ov
    return best if best is not None else 0


def main():
    path = sys.argv[1]
    num = int(sys.argv[2]) if len(sys.argv) > 2 else -1

    audio, _ = read_wav_mono_f32(path)  # sherpa/whisper both want 16k mono
    turns = diarize(audio, num)

    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(path, language="en", condition_on_previous_text=False)

    out = []
    for seg in segments:
        spk = speaker_for(seg.start, seg.end, turns)
        out.append(
            {"speaker": int(spk), "start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
        )

    labeled = " ".join(f"[S{s['speaker']}] {s['text']}" for s in out)
    print(json.dumps({"segments": out, "text": labeled, "num_speakers": len({s["speaker"] for s in out})}))


if __name__ == "__main__":
    main()

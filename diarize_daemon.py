#!/usr/bin/env python3
"""Warm diarized-transcription daemon for BFS meeting capture.

Holds the sherpa-onnx diarization models + faster-whisper warm in RAM and serves
speaker-labeled transcription over HTTP on 127.0.0.1:7980.

  POST /diarize  { "path": "<wav>", "num_speakers": int (optional, <=0 = auto) }
    -> { "segments": [{speaker,start,end,text}], "text": "[S0].. [S1]..",
         "num_speakers": int }
  GET  /health -> { "status": "ok" }

Uses freely-downloadable ONNX models (no HuggingFace gate).
"""
import json
import os
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import sherpa_onnx
from faster_whisper import WhisperModel

PORT = 7980
HERE = "/home/pbrown/s2t/diar-models"
SEG = f"{HERE}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB = os.environ.get("DIAR_EMB", f"{HERE}/titanet.onnx")
WHISPER_MODEL = os.environ.get("DIAR_WHISPER", "small.en")
THRESHOLD = float(os.environ.get("DIAR_THRESHOLD", "0.5"))

whisper = None


def read_wav_mono_f32(path):
    with wave.open(path, "rb") as w:
        n, ch = w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a


def make_diarizer(num_speakers):
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
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)


def speaker_for(start, end, turns):
    best, best_ov = 0, 0.0
    for ts, te, spk in turns:
        ov = max(0.0, min(end, te) - max(start, ts))
        if ov > best_ov:
            best, best_ov = spk, ov
    return best


def diarize_transcribe(path, num_speakers, initial_prompt=None):
    audio = read_wav_mono_f32(path)
    sd = make_diarizer(num_speakers)  # cheap to build; models are cached by sherpa
    turns = [(t.start, t.end, t.speaker) for t in sd.process(audio).sort_by_start_time()]
    segments, _ = whisper.transcribe(path, language="en", condition_on_previous_text=False, initial_prompt=initial_prompt)
    out = [
        {
            "speaker": int(speaker_for(s.start, s.end, turns)),
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text.strip(),
        }
        for s in segments
    ]
    text = " ".join(f"[S{s['speaker']}] {s['text']}" for s in out)
    return {"segments": out, "text": text, "num_speakers": len({s["speaker"] for s in out})}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self._respond(200 if self.path == "/health" else 404, {"status": "ok"})

    def do_POST(self):
        if self.path != "/diarize":
            self._respond(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        path = body.get("path", "")
        if not Path(path).exists():
            self._respond(400, {"error": f"file not found: {path}"})
            return
        try:
            self._respond(200, diarize_transcribe(path, int(body.get("num_speakers", -1)), (body.get("initial_prompt") or "").strip() or None))
        except Exception as e:  # noqa: BLE001
            self._respond(500, {"error": str(e)})

    def _respond(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print("Loading whisper + diarization models...", flush=True)
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print(f"Diarize daemon ready on 127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

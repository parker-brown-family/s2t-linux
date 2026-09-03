#!/usr/bin/env python3
"""S2T warm daemon — holds a Whisper model in RAM, serves transcription over HTTP on 127.0.0.1:7979.

Model is env-selectable via S2T_WHISPER (default: small.en). BFS meetings are
English, so the .en variants are more accurate than multilingual at the same
size; small.en is the accuracy/cost sweet spot on CPU for offline (batch)
transcription. See piper/SPEC/07-model-cost-accuracy.md.

Optional `initial_prompt` in the /transcribe body biases recognition toward
domain vocabulary (piper's persona hotwords) — near-zero-cost accuracy (#7)."""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from faster_whisper import WhisperModel

PORT = 7979
WHISPER_MODEL = os.environ.get("S2T_WHISPER", "small.en")
model = None


def load_model():
    global model
    print(f"Loading Whisper {WHISPER_MODEL} model...", flush=True)
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print(f"Model ready. Listening on 127.0.0.1:{PORT}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/transcribe":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            audio_path = body.get("path", "")
            # Optional domain-bias prompt (#7); empty/missing -> no prompt.
            prompt = (body.get("initial_prompt") or "").strip() or None
            if not Path(audio_path).exists():
                self._respond(400, {"error": f"file not found: {audio_path}"})
                return
            segments, _ = model.transcribe(
                audio_path,
                language="en",
                condition_on_previous_text=False,
                initial_prompt=prompt,
            )
            text = " ".join(seg.text.strip() for seg in segments)
            self._respond(200, {"text": text})
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    load_model()
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Daemon stopped.")

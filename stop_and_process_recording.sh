#!/bin/bash

# Stop recording
kill $(cat $HOME/s2t/tmp/recording_pid)

AUDIO_FILE="$HOME/s2t/tmp/recording.wav"
TEXT_FILE="$HOME/s2t/tmp/recording.txt"

# Try warm daemon first (sub-second). Fall back to direct faster-whisper+tiny (~2s).
if curl -sf --max-time 0.5 http://127.0.0.1:7979/health > /dev/null 2>&1; then
    TEXT=$(curl -sf --max-time 15 -X POST http://127.0.0.1:7979/transcribe \
        -H "Content-Type: application/json" \
        -d "{\"path\":\"$AUDIO_FILE\"}" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])" 2>/dev/null)
    echo "$TEXT" > "$TEXT_FILE"
else
    $HOME/env_sandbox/bin/python3 $HOME/s2t/transcribe.py "$AUDIO_FILE" "$TEXT_FILE"
fi

# Apply phrase expansions (phrases.json)
$HOME/env_sandbox/bin/python3 $HOME/s2t/expand_phrases.py "$TEXT_FILE"

# Copy transcription to clipboard
xclip -selection clipboard < "$TEXT_FILE"

# Notify
notify-send "Transcription Complete" "$(cat "$TEXT_FILE")"

# Ensure the clipboard has time to update
sleep 0.1

# Simulate the paste action
xdotool key ctrl+v

# Clean up
rm -rf $HOME/s2t/tmp/

#!/bin/bash
# Make test clips: 5 built-in macOS voices x 3 different sentences, 16 kHz mono WAV.
set -e
mkdir -p clips
SENTENCES=(
  "The quick brown fox jumps over the lazy dog near the river bank."
  "Please call Stella and ask her to bring these things from the store."
  "Our research team measures speech quality at different step counts."
)
for VOICE in Daniel Karen Rishi Tessa Moira; do
  for i in 0 1 2; do
    say -v "$VOICE" -o "clips/${VOICE}_$((i+1)).wav" --file-format=WAVE --data-format=LEI16@16000 "${SENTENCES[$i]}"
  done
done
ls clips | wc -l | xargs echo "clips made:"

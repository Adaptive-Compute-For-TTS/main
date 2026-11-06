#!/bin/bash
# Add a real recording (Voice Memos .m4a, QuickTime .mov/.m4a, .mp3, .wav ...) to clips/
# as 16 kHz mono WAV, named for the sanity check: <Name>_<n>.wav
# Keep the originals in recordings/ (git-ignored); converted copies go to clips/.
#
# Usage: ./add_recording.sh recordings/Atharv_1.m4a Atharv_1
set -euo pipefail
if [ $# -ne 2 ]; then
  echo "usage: $0 <audio-file> <Name_n>   e.g. $0 recordings/Atharv_1.m4a Atharv_1" >&2
  exit 1
fi
IN="$1"; NAME="$2"
if [[ ! "$NAME" =~ ^[A-Za-z]+_[0-9]+$ ]]; then
  echo "name must look like Atharv_1 (a name, underscore, number)" >&2
  exit 1
fi
mkdir -p clips
afconvert -f WAVE -d LEI16@16000 -c 1 "$IN" "clips/$NAME.wav"
echo "added clips/$NAME.wav"

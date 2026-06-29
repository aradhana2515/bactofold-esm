#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
curl -L "https://dbarchive.biosciencedbc.jp/data/esol/LATEST/esol.zip" -o data/raw/esol.zip
ls -lh data/raw/esol.zip

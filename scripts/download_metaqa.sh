#!/usr/bin/env bash
# Download MetaQA (kb.txt + vanilla questions) into data/raw/MetaQA and check the layout.
# Source: Google Drive folder linked from https://github.com/yuyuz/MetaQA
set -uo pipefail
cd "$(dirname "$0")/.."
DEST=data/raw/MetaQA
URL="https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE?resourcekey=0-kdv6ho5KcpEXdI2aUdLn_g"

check() {
  local ok=1
  for f in kb.txt {1,2,3}-hop/vanilla/qa_{train,dev,test}.txt; do
    if [ ! -s "$DEST/$f" ]; then echo "missing: $DEST/$f"; ok=0; fi
  done
  return $((1 - ok))
}

if check >/dev/null 2>&1; then echo "MetaQA already in place at $DEST"; exit 0; fi

mkdir -p data/raw/_metaqa_download
if command -v gdown >/dev/null 2>&1; then
  echo "Trying gdown (the folder is large; audio files are not needed)..."
  # --remaining-ok exists only in older gdown versions
  gdown --folder "$URL" -O data/raw/_metaqa_download --remaining-ok 2>/dev/null \
    || gdown --folder "$URL" -O data/raw/_metaqa_download \
    || echo "gdown did not finish"
  SRC=$(find data/raw/_metaqa_download -name kb.txt -printf '%h\n' | head -1)
  if [ -n "$SRC" ]; then
    mkdir -p "$DEST"
    cp "$SRC/kb.txt" "$DEST/"
    for h in 1 2 3; do
      if [ -d "$SRC/$h-hop/vanilla" ]; then
        mkdir -p "$DEST/$h-hop/vanilla" && cp "$SRC/$h-hop/vanilla/"qa_*.txt "$DEST/$h-hop/vanilla/"
      fi
      # some mirrors ship the vanilla folder zipped
      for z in "$SRC/$h-hop/"*.zip; do [ -f "$z" ] && unzip -o -q "$z" -d "$DEST/$h-hop/"; done
    done
  fi
else
  echo "gdown not found (pip install gdown)."
fi

if check; then
  echo "MetaQA ready at $DEST"
else
  cat <<EOF

MetaQA is not complete. Download by hand from:
  $URL
and place these files:
  $DEST/kb.txt
  $DEST/{1,2,3}-hop/vanilla/qa_{train,dev,test}.txt
Then run this script again to check the layout.
EOF
  exit 1
fi

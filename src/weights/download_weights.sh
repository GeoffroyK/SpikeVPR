#!/usr/bin/env bash
# Download the SpikeVPR model weights into this folder.
#
# Set BASE_URL to the release location that hosts the .pth files (e.g. a GitHub
# release, Zenodo record or HuggingFace repo), then run ./download_weights.sh.
# Checksums are verified against SHA256SUMS.txt.
set -euo pipefail
cd "$(dirname "$0")"

BASE_URL="${SPIKEVPR_WEIGHTS_URL:-https://REPLACE_ME/spikevpr/weights}"

FILES=(
  sew_resnet18_brisbane.pth
  sew_resnet34_brisbane.pth
  sew_resnet18_nsavp.pth
  sew_resnet34_nsavp.pth
  sew_resnet34_nyc.pth
  netvlad_weights.pth
  wpca_weights.pth
)

if [[ "$BASE_URL" == *REPLACE_ME* ]]; then
  echo "Set the download location first, e.g.:"
  echo "  SPIKEVPR_WEIGHTS_URL=https://your-host/spikevpr/weights ./download_weights.sh"
  exit 1
fi

for f in "${FILES[@]}"; do
  if [[ -f "$f" ]]; then
    echo "exists  $f"
  else
    echo "fetch   $f"
    curl -L --fail -o "$f" "$BASE_URL/$f"
  fi
done

echo "Verifying checksums..."
sha256sum -c SHA256SUMS.txt
echo "Done."

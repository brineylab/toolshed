#!/usr/bin/env bash
# Download the HyPLM-Flash weights (two shards) from the GitHub Release, merge them and verify the sha256.
# The release is public, so the fetch needs no token; --retry rides out the CDN's transient errors.
set -euo pipefail
cd "$(dirname "$0")"
REPO=Tencent-Hunyuan/Hyra-results
TAG=hyplm-flash-weights
SHA=ce7dcd4fc473dc334d869718a9a31149573893795db03c45efeb396e8b180ab1
for p in part0 part1; do
  curl -fL --retry 5 --retry-all-errors -o "model.safetensors.$p" "https://github.com/$REPO/releases/download/$TAG/model.safetensors.$p"
done
cat model.safetensors.part0 model.safetensors.part1 > model.safetensors
echo "$SHA  model.safetensors" | sha256sum -c -
rm -f model.safetensors.part0 model.safetensors.part1
printf '%s\n' "Tencent-Hunyuan/Hyra-results, release $TAG (AI4Science/hyplm_flash)" \
  "model.safetensors sha256 $SHA" > .weights_source
echo "Weights ready: $(pwd)/model.safetensors"

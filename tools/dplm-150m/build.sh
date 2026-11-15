#!/usr/bin/env bash
# The whole build in one quiet RUN, so the build log stays a handful of lines: every step
# is silent on success and says why it failed. The environment (pinned packages, quiet),
# then the weights (pinned, sha256-verified, fetched into /opt/model by fetch_weights.py,
# silent on success), then the committed files laid into the image: the vendored modeling
# code under /opt/src, the wrapper on PATH, and the upstream license beside the weights.
set -euo pipefail
pip install -q --no-cache-dir --break-system-packages transformers==4.51.3 tokenizers==0.21.1 huggingface_hub==0.30.2 safetensors==0.5.3 numpy==2.5.4
python3 -I /incoming/fetch_weights.py /opt/model
cp -r /incoming/src /opt/src
install -m 755 /incoming/bin/dplm /usr/local/bin/dplm
cp /incoming/LICENSE /opt/model/LICENSE
rm -rf /incoming

#!/usr/bin/env bash
# The whole build in one RUN, one named line per stage, so a build that dies says where.
# requirements.txt is the closed set resolved for this image's Python 3.12 (transformers
# 4.51.3, tokenizers 0.21.1, huggingface_hub 0.30.2, safetensors 0.5.3, numpy 2.5.4, and
# their transitives), each wheel sha256-pinned; --no-deps keeps pip from resolving past
# it. The weights (pinned, sha256-verified, fetched into /opt/model by fetch_weights.py)
# follow, then the committed files are laid into the image: the vendored modeling code
# under /opt/src, the wrapper on PATH, and the upstream license beside the weights.
set -euo pipefail
echo "dplm build: installing the pinned packages"
pip install -q --no-cache-dir --no-deps --require-hashes --break-system-packages -r /incoming/requirements.txt
python3 -I -c "import transformers, tokenizers, huggingface_hub, safetensors, numpy; assert (transformers.__version__, tokenizers.__version__, huggingface_hub.__version__, safetensors.__version__, numpy.__version__) == ('4.51.3', '0.21.1', '0.30.2', '0.5.3', '2.5.4'), 'the packages did not install as pinned'"
echo "dplm build: fetching the pinned weights"
python3 -I /incoming/fetch_weights.py /opt/model
echo "dplm build: laying in the committed files"
cp -r /incoming/src /opt/src
install -m 755 /incoming/bin/dplm /usr/local/bin/dplm
cp /incoming/LICENSE /opt/model/LICENSE
rm -rf /incoming

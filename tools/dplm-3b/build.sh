#!/usr/bin/env bash
# The environment half of dplm-150m's build, verbatim; the weights half moved
# into the Containerfile, one RUN per shard. The four fp32 shards total
# 11.4 GB: fetching them in this RUN would make one layer that nears the
# registry's layer-size limit, so build.sh stops after the environment and
# the committed files, and the Containerfile fetches each shard in its own
# RUN. The environment is vendored: the wheels under wheels/ are the closed
# set resolved for this image's Python 3.12 (transformers 4.51.3, tokenizers
# 0.21.1, huggingface_hub 0.30.2, safetensors 0.5.3, numpy 2.5.4, and their
# transitives), sha256-pinned in wheels/SHA256SUMS and installed with
# --no-index, so pip needs no package index. The committed files are laid into
# the image: the vendored modeling code under /opt/src, the wrapper on PATH,
# and the upstream license beside the weights (/opt/model, created here so the
# shard fetches have their directory).
set -euo pipefail
echo "dplm build: verifying the vendored wheels"
(cd /incoming/wheels && sha256sum --quiet -c SHA256SUMS)
echo "dplm build: installing the pinned packages offline"
pip install -q --no-index --no-cache-dir --no-deps --break-system-packages /incoming/wheels/*.whl
python3 -I -c "import transformers, tokenizers, huggingface_hub, safetensors, numpy; assert (transformers.__version__, tokenizers.__version__, huggingface_hub.__version__, safetensors.__version__, numpy.__version__) == ('4.51.3', '0.21.1', '0.30.2', '0.5.3', '2.5.4'), 'the vendored wheels did not install as pinned'"
echo "dplm build: laying in the committed files"
cp -r /incoming/src /opt/src
install -m 755 /incoming/bin/dplm /usr/local/bin/dplm
mkdir -p /opt/model
cp /incoming/LICENSE /opt/model/LICENSE

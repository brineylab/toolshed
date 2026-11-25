set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
test -f /opt/model/LICENSE
test -f /opt/model/config.json
test -f /opt/model/model.safetensors
test -f /opt/model/vocab.txt
antiberta2 pll --heavy heavy.fasta --light light.fasta --out "$out/pll.tsv"
antiberta2 embed --heavy heavy.fasta --light light.fasta --out "$out/emb.npz"
antiberta2 fill --heavy masked-heavy.fasta --light masked-light.fasta --out "$out/filled.fasta"
antiberta2 matrix --heavy heavy.fasta --light light.fasta --out "$out/matrix.npz"
python3 -I check.py "$out"

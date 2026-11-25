set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
test -f /opt/AntiBERTy/LICENSE
test -f /opt/AntiBERTy/config.json
test -f /opt/AntiBERTy/model.safetensors
test -f /opt/AntiBERTy/vocab.txt
antiberty pll --in input.fasta --out "$out/pll.tsv" --normalize full-vocab
antiberty embed --in input.fasta --out "$out/emb.npz"
antiberty fill --in masked.fasta --out "$out/filled.fasta"
antiberty matrix --in input.fasta --out "$out/matrix.npz"
python3 -I check.py "$out"

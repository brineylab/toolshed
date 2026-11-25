set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
test -f /opt/model/LICENSE
test -f /opt/model/config.json
test -f /opt/model/model.safetensors
test -f /opt/model/vocab.txt
igbert-unpaired pll --in sequences.fasta --out "$out/pll.tsv"
igbert-unpaired embed --in sequences.fasta --out "$out/emb.npz"
igbert-unpaired fill --in masked.fasta --out "$out/filled.fasta"
igbert-unpaired matrix --in sequences.fasta --out "$out/matrix.npz"
python3 -I check.py "$out"

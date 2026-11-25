set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
test -f /opt/ablang/LICENSE
test -f /opt/ablang/amodel.pt
test -f /opt/ablang/hparams.json
test -f /opt/ablang/vocab.json
command="ablang1-${ABLANG1_CHAIN}"
"$command" pll --in sequences.fasta --out "$out/pll.tsv"
"$command" embed --in sequences.fasta --out "$out/emb.npz"
"$command" fill --in masked.fasta --out "$out/filled.fasta"
"$command" matrix --in sequences.fasta --out "$out/matrix.npz"
python3 -I check.py "$out"

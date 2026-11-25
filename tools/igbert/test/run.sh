set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
test -f /opt/model/LICENSE
test -f /opt/model/config.json
test -f /opt/model/model.safetensors
test -f /opt/model/vocab.txt
igbert pll --heavy heavy.fasta --light light.fasta --out "$out/pll.tsv"
igbert embed --heavy heavy.fasta --light light.fasta --out "$out/emb.npz"
igbert fill --heavy masked-heavy.fasta --light masked-light.fasta --out "$out/filled.fasta"
igbert matrix --heavy heavy.fasta --light light.fasta --out "$out/matrix.npz"
python3 -I check.py "$out"

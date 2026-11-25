set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
test -f /opt/ABLANG-paired/LICENSE
test -f /opt/ABLANG-paired/LICENSE.weights
test -f /opt/ABLANG-paired/model.pt
ablang2 pll --heavy heavy.fasta --light light.fasta --out "$out/pll.tsv"
ablang2 embed --heavy heavy.fasta --light light.fasta --out "$out/emb.npz"
ablang2 fill --heavy masked-heavy.fasta --light masked-light.fasta --out "$out/filled.fasta"
ablang2 matrix --heavy heavy.fasta --light light.fasta --out "$out/matrix.npz"
python3 -I check.py "$out"

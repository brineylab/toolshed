set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
# the license files the skill says ride beside the weights (Gotchas)
test -f /opt/model/LICENSE.md && test -f /opt/model/THIRD_PARTY_NOTICE.md
esmc pll --in seqs.fasta --out "$out/pll.tsv"
esmc embed --in seqs.fasta --out "$out/emb.npz"
esmc fill --in masked.fasta --out "$out/filled.fasta"
esmc contacts --in contacts.fasta --out "$out/contacts.npz"
python3 -I check.py "$out/pll.tsv" "$out/emb.npz" "$out/filled.fasta" "$out/contacts.npz"

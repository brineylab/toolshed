set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
# the license the skill says rides beside the weights (Gotchas)
test -f /opt/model/LICENSE
esm2 pll --in seqs.fasta --out "$out/pll.tsv"
esm2 embed --in seqs.fasta --out "$out/emb.npz"
esm2 fill --in masked.fasta --out "$out/filled.fasta"
esm2 contacts --in contacts.fasta --out "$out/contacts.npz"
python3 -I check.py "$out/pll.tsv" "$out/emb.npz" "$out/filled.fasta" "$out/contacts.npz"

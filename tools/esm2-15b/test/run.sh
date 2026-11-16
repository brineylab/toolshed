set -euo pipefail
cd "$(dirname "$0")"
# the first command verifies every group shard's value pin against the manifest
# (hashing the ~30 GB of bf16 once per machine, cached by a tempdir marker), then
# loads the model in bf16; nothing is assembled or written outside mktemp -d
out=$(mktemp -d)
# the license the skill says rides beside the weights (Gotchas)
test -f /opt/model/LICENSE
esm2 pll --in seqs.fasta --out "$out/pll.tsv"
esm2 embed --in seqs.fasta --out "$out/emb.npz"
esm2 fill --in masked.fasta --out "$out/filled.fasta"
esm2 contacts --in contacts.fasta --out "$out/contacts.npz"
python3 -I check.py "$out/pll.tsv" "$out/emb.npz" "$out/filled.fasta" "$out/contacts.npz"

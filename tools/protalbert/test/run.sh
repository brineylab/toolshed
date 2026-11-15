set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
protalbert contacts --in seqs.fasta --out "$out/contacts.npz"
python3 -I check.py "$out/contacts.npz" ref.npz

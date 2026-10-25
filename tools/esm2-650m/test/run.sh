set -euo pipefail
cd "$(dirname "$0")"
esm2 pll --in seqs.fasta --out /tmp/pll.tsv
esm2 embed --in seqs.fasta --out /tmp/emb.npz
esm2 fill --in masked.fasta --out /tmp/filled.fasta
python3 check.py /tmp/pll.tsv /tmp/emb.npz /tmp/filled.fasta

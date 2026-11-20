#!/usr/bin/env bash
# tmol's test. The fixture is water-stripped 1UBQ (76 residues; the
# record's 58 waters removed), expanded to 26 copies: tmol's packer
# processes a pose stack whose poses hold at most 256 blocks in chunks
# of 25 poses, so 26 copies crosses the boundary and the test exercises
# the chunked path (25 poses, then 1). Every documented command runs: the
# four verbs on the directory batch, a seeded reproducibility pair, the
# chaining invariant (tmol score on an optimizer verb's outdir
# reproduces its scores.tsv), and the library way (one python3 -I
# script that relaxes then scores, in one process). check.py verifies
# the answers against values pinned by the campaign's measurement run.
set -euo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d /tmp/tmol.XXXXXX)
trap 'rm -rf "$T"' EXIT

mkdir -p "$T/in"
for i in $(seq -w 1 26); do
    cp 1UBQ.pdb "$T/in/pose_$i.pdb"
done

tmol score "$T/in" "$T/scores.tsv"
tmol minimize "$T/in" "$T/min"
tmol pack "$T/in" "$T/pack-a" --seed 1
tmol pack "$T/in" "$T/pack-b" --seed 1
tmol relax "$T/in" "$T/relax"

# the chaining invariant: scoring a verb's outdir reproduces its scores.tsv
tmol score "$T/min" "$T/score-min.tsv"

# the library way: relax then score, one process, under python3 -I
cat > "$T/relax_score.py" <<'PY'
import sys

import torch
from tmol import (
    beta2016_score_function,
    fast_relax,
    file_from_pose_stack,
    pose_stack_from_file,
)

source, output = sys.argv[1], sys.argv[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

pose = pose_stack_from_file(source, device, packer_seed=0)
sfxn = beta2016_score_function(device)
scorer = sfxn.render_whole_pose_scoring_module(pose)
with torch.no_grad():
    before = scorer(pose.coords).tolist()

relaxed = fast_relax(pose, sfxn)

# Re-render the scorer on the relaxed pose: its atom layout can change,
# and the old scorer is bound to the old layout.
scorer = sfxn.render_whole_pose_scoring_module(relaxed)
with torch.no_grad():
    after = scorer(relaxed.coords).tolist()

file_from_pose_stack(relaxed, output)
print(f"before {before[0]:.6f}")
print(f"after {after[0]:.6f}")
PY
python3 -I "$T/relax_score.py" "$T/in/pose_01.pdb" "$T/demo-relaxed.pdb" \
    > "$T/demo.txt"

python3 check.py "$T"

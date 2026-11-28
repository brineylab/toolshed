set -euo pipefail
cd "$(dirname "$0")"
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "[$(stamp)] $(hostname): $(nvidia-smi -L 2>/dev/null || echo 'no gpu visible')"
out=$(mktemp -d)
inputs=$out/inputs
mkdir "$inputs"
cp chignolin.yaml chignolin.a3m "$inputs/"
# No HOME in the cluster job: keeps every runtime write (Triton kernels, RDKit,
# ColabFold misses) off the image and under the scratch dir, so the test is the
# same as an offline caller's.
export HOME="$out/home"
export TRITON_CACHE_DIR="$out/triton"
mkdir "$HOME" "$TRITON_CACHE_DIR"
# Pyxis passes the caller's environment through. The CLI must use the baked-in
# checkpoint cache even when the caller asks Hugging Face to look elsewhere.
export HF_HOME=/tmp/nowhere HF_HUB_CACHE=/tmp/nowhere
python -I - <<'PY'
import json
from pathlib import Path
import bioemu2
from bioemu2.checkpoints import get_checkpoint_files
print("bioemu2:", bioemu2.__file__)
f = get_checkpoint_files("bioemu2-preview", cache_dir="/opt/bioemu2/weights")
print("checkpoint from image cache:", f)
assert str(f.checkpoint_path).startswith("/opt/bioemu2/weights/"), f
assert f.checkpoint_path.stat().st_size > 1_000_000_000
assert Path("/opt/bioemu2/licenses/upstream-LICENSE").stat().st_size > 1000
import hashlib
for entry in Path("/opt/bioemu2/msa-cache").glob("*.json"):
    d = json.loads(entry.read_text())
    assert d["sha256"] == hashlib.sha256(json.dumps(d["a3ms"]).encode()).hexdigest(), entry
print("licences, checkpoint and cached MSA checksums verified")
PY
# 20 denoising steps rather than the default 200: the test's fixture is tiny, its
# assertions are shapes/topology/physicality counts, and the checkpoint load and
# first-use Triton compiles come out of the same 30-minute budget on any GPU.
steps=(--num-steps 20 --seed 0)
echo "[$(stamp)] 1/4 predict-structure, system YAML + local A3M (--msa)"
bioemu2 predict-structure --system "$inputs/chignolin.yaml" --msa "$inputs/chignolin.a3m" \
  --num-samples 1 "${steps[@]}" --out "$out/predict"
echo "[$(stamp)] 2/4 sample-equilibrium, cached ColabFold MSA"
bioemu2 sample-equilibrium --system "$inputs/chignolin.yaml" --num-samples 2 \
  "${steps[@]}" --out "$out/ensemble"
echo "[$(stamp)] 3/4 propagate one endpoint per sampled frame, 1 ns"
bioemu2 propagate --x0-cif "$out/ensemble/topology.cif" --x0-xtc "$out/ensemble/samples.xtc" \
  --lag-time-ns 1 "${steps[@]}" --out "$out/propagate"
echo "[$(stamp)] 4/4 predict-structure --pdb-id 1AKE (monomeric protein + AP5 ligand)"
bioemu2 predict-structure --pdb-id 1AKE --num-samples 1 "${steps[@]}" --out "$out/pdb"
echo "[$(stamp)] checking outputs"
python3 -I check.py "$out"
echo "[$(stamp)] all checks passed"
if [ -n "${FLYLOFT_OUT:-}" ] && [ -d "${FLYLOFT_OUT}" ]; then
  cp "$out"/predict/physicality.json "${FLYLOFT_OUT}"/predict-physicality.json
  cp "$out"/pdb/physicality.json "${FLYLOFT_OUT}"/pdb-physicality.json
  python3 -I - "$out" "${FLYLOFT_OUT}/summary.json" <<'PY'
import json, subprocess, sys
from pathlib import Path
out, summary = sys.argv[1:]
report = {"directory": out,
          "sizes": {p.name: p.stat().st_size for p in Path(out).glob("*/*.cif")},
          "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                                capture_output=True, text=True).stdout}
Path(summary).write_text(json.dumps(report, indent=2))
PY
fi

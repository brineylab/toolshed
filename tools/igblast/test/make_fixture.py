#!/usr/bin/env python3
"""Builds the test fixture: productive rearrangements joined out of IMGT germline.

Each sequence is V-REGION + N + (D) + N + J-REGION, with N-region lengths chosen so the
J lands in the coding frame its human_gl.aux line gives, and with no stop codon anywhere;
eight of the fourteen carry 3-6 somatic substitutions at positions drawn from across V
(framework and CDR alike). The V, D and J genes are therefore known by construction, and
so is the V identity implied by the substitutions, which is what test/run.sh checks
IgBLAST against. It is run once, by hand, and its output committed:

    python3 make_fixture.py <dir with imgt_human_ig_v.fa, imgt_human_ig_d.fa, imgt_human_ig_j.fa> <human_gl.aux>

<written from the germline the image builds: the combined, edit_imgt_file.pl-processed
IMGT human IG V, D and J files>
"""
import pathlib
import random
import sys

STOPS = {"TAA", "TAG", "TGA"}

# sequence_id, V, D, J, substitutions at positions drawn from across V, rng seed
HEAVY = [
    ("H01", "IGHV1-18*01", "IGHD3-22*01", "IGHJ4*02", 0, 101),
    ("H02", "IGHV3-23*01", "IGHD6-19*01", "IGHJ6*02", 4, 102),
    ("H03", "IGHV4-59*01", "IGHD3-22*01", "IGHJ4*02", 6, 103),
    ("H04", "IGHV3-23*01", "IGHD3-22*01", "IGHJ4*02", 0, 104),
    ("H05", "IGHV1-18*01", "IGHD6-19*01", "IGHJ6*02", 3, 105),
    ("H06", "IGHV4-59*01", "IGHD6-19*01", "IGHJ4*02", 5, 106),
]
KAPPA = [
    ("K01", "IGKV3-20*01", None, "IGKJ1*01", 0, 201),
    ("K02", "IGKV1-39*01", None, "IGKJ2*01", 4, 202),
    ("K03", "IGKV3-20*01", None, "IGKJ2*01", 0, 203),
    ("K04", "IGKV1-39*01", None, "IGKJ1*01", 3, 204),
]
LAMBDA = [
    ("L01", "IGLV2-14*01", None, "IGLJ3*02", 0, 301),
    ("L02", "IGLV1-51*01", None, "IGLJ2*01", 4, 302),
    ("L03", "IGLV2-14*01", None, "IGLJ2*01", 0, 303),
    ("L04", "IGLV1-51*01", None, "IGLJ3*02", 3, 304),
]


def read_fasta(path):
    seqs, name, buf = {}, None, []
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf).upper()
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf).upper()
    return seqs


def read_aux(path):
    aux = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split()
        aux[f[0]] = int(f[1])  # first coding frame start position, 0-based
    return aux


def has_stop(seq):
    """A stop codon in the rearrangement's own reading frame, which runs from the first
    base of V, not from the first base of the junction."""
    return any(seq[i:i + 3] in STOPS for i in range(0, len(seq) - 2, 3))


def make(spec, v_seq, d_seq, j_seq, j_frame):
    """V + N + (D) + N + J, J in the frame human_gl.aux gives, no stop codons."""
    sid, v, d, j, n_sub, seed = spec
    rng = random.Random(seed)
    for _ in range(1000):  # redraw N regions until the join is productive
        n1 = "".join(rng.choice("ACGT") for _ in range(rng.randint(4, 10)))
        n2 = "".join(rng.choice("ACGT") for _ in range(rng.randint(3, 9)))
        # pad N2 so the J's first coding-frame base sits on a codon boundary
        need = (len(v_seq) + len(n1) + len(d_seq) + len(n2) + j_frame) % 3
        if need:
            n2 += "".join(rng.choice("ACGT") for _ in range(3 - need))
        seq = v_seq + n1 + d_seq + n2 + j_seq
        if has_stop(seq):
            continue
        # somatic substitutions at positions drawn from across V (framework and CDR
        # alike), away from the 3' end
        for _ in range(n_sub):
            for _ in range(100):
                p = rng.randrange(9, len(v_seq) - 45)
                alt = rng.choice([b for b in "ACGT" if b != seq[p]])
                trial = seq[:p] + alt + seq[p + 1:]
                if not has_stop(trial):
                    seq = trial
                    break
        if has_stop(seq):
            continue
        return sid, seq
    raise RuntimeError(f"could not build a productive join for {sid}")


def main():
    germline_dir, aux_file, out_dir = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    v = read_fasta(f"{germline_dir}/imgt_human_ig_v.fa")
    d = read_fasta(f"{germline_dir}/imgt_human_ig_d.fa")
    j = read_fasta(f"{germline_dir}/imgt_human_ig_j.fa")
    aux = read_aux(aux_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, specs in [("heavy", HEAVY), ("kappa", KAPPA), ("lambda", LAMBDA)]:
        with open(out_dir / f"{group}.fa", "w") as fh:
            for spec in specs:
                sid, seq = make(spec, v[spec[1]], d.get(spec[2], ""), j[spec[3]], aux[spec[3]])
                fh.write(f">{sid} {spec[1]}+{spec[2] or ''}+{spec[3]} len={len(seq)}\n")
                for i in range(0, len(seq), 60):
                    fh.write(seq[i:i + 60] + "\n")
                v_seq = v[spec[1]]
                subs = sum(1 for i in range(len(v_seq)) if seq[i] != v_seq[i])
                v_identity = 100.0 * (len(v_seq) - subs) / len(v_seq)
                rows.append((sid, spec[1], spec[2] or "", spec[3], f"{v_identity:.3f}"))
    with open(out_dir / "expected.tsv", "w") as fh:
        fh.write("sequence_id\tv_call\td_call\tj_call\tproductive\tv_identity\n")
        for sid, vc, dc, jc, vid in rows:
            fh.write(f"{sid}\t{vc}\t{dc}\t{jc}\tT\t{vid}\n")
    print(f"wrote {len(rows)} sequences to {out_dir}")


if __name__ == "__main__":
    main()

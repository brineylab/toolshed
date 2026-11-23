#!/bin/bash
# IgBLAST's test: annotates fourteen known heavy, kappa and lambda chains — productive joins of IMGT germline built by
# make_fixture.py, four with framework substitutions — and checks the AIRR output calls the V, D and J genes they were
# built from, that every chain is productive with a stop-free junction, and that light chains get no D call.
set -euo pipefail
cd "$(dirname "$0")"

GL=/usr/local/share/igblast/database/imgt_human_ig

igblastn -version | grep -F 'igblastn: 1.22.0'

work=$(mktemp -d /tmp/igblast-test.XXXXXX)
trap 'rm -rf "$work"' EXIT

for group in heavy kappa lambda; do
    igblastn \
        -germline_db_V "$GL/imgt_human_ig_v.fa" \
        -germline_db_D "$GL/imgt_human_ig_d.fa" \
        -germline_db_J "$GL/imgt_human_ig_j.fa" \
        -organism human -ig_seqtype Ig \
        -auxiliary_data /usr/local/share/igblast/optional_file/human_gl.aux \
        -query "fixture/$group.fa" -out "$work/$group.tsv" -outfmt 19 -num_threads 2
done

for group in heavy kappa lambda; do
    case $group in heavy) prefix=H;; kappa) prefix=K;; lambda) prefix=L;; esac
    awk -v group="$group" -v prefix="$prefix" -v tsv="$work/$group.tsv" 'BEGIN {
        FS = "\t"
        n = 0
        while ((getline line < "fixture/expected.tsv") > 0) {
            if (n++ == 0) continue
            split(line, f, "\t")
            if (substr(f[1], 1, 1) != prefix) continue
            ev[f[1]] = f[2]; ed[f[1]] = f[3]; ej[f[1]] = f[4]
            want[f[1]] = 1
        }
        rows = 0; bad = 0
        while ((getline line < tsv) > 0) {
            if (rows++ == 0) {
                split(line, h, "\t")
                for (i in h) col[h[i]] = i
                continue
            }
            split(line, f, "\t")
            id = f[col["sequence_id"]]
            if (!(id in want)) { printf "%s: unexpected sequence %s\n", group, id; bad++; continue }
            ok = 1
            m = split(f[col["v_call"]], vs, ","); found = 0
            for (i = 1; i <= m; i++) if (vs[i] == ev[id]) found = 1
            if (!found) { printf "%s %s: v_call %s, expected %s\n", group, id, f[col["v_call"]], ev[id]; ok = 0 }
            if (ed[id] == "") {
                if (f[col["d_call"]] != "") { printf "%s %s: light chain has d_call %s\n", group, id, f[col["d_call"]]; ok = 0 }
            } else if (f[col["d_call"]] != ed[id]) {
                printf "%s %s: d_call %s, expected %s\n", group, id, f[col["d_call"]], ed[id]; ok = 0
            }
            m = split(f[col["j_call"]], js, ","); found = 0
            for (i = 1; i <= m; i++) if (js[i] == ej[id]) found = 1
            if (!found) { printf "%s %s: j_call %s, expected %s\n", group, id, f[col["j_call"]], ej[id]; ok = 0 }
            if (f[col["productive"]] != "T") { printf "%s %s: productive %s, expected T\n", group, id, f[col["productive"]]; ok = 0 }
            if (f[col["junction_aa"]] == "" || f[col["junction_aa"]] ~ /\*/) {
                printf "%s %s: junction_aa %s\n", group, id, f[col["junction_aa"]]; ok = 0
            }
            if (ok) got[id] = 1; else bad++
        }
        for (id in want) if (!(id in got)) { printf "%s: %s missing from the AIRR output\n", group, id; bad++ }
        printf "%s: %d sequences, %d failures\n", group, rows - 1, bad
        exit bad ? 1 : 0
    }'
done

echo "igblast test passed"

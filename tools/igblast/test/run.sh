#!/bin/bash
# IgBLAST's test: annotates fourteen known heavy, kappa and lambda chains — productive joins of IMGT germline built by
# make_fixture.py, eight of them carrying 3-6 substitutions at positions drawn from across V (framework and CDR alike) —
# and checks the AIRR output (-outfmt 19) calls the V, D and J genes they were built from, with the v_identity those
# substitutions imply, that every chain is productive with a stop-free junction, and that light chains get no D call;
# then reruns the heavy group with -outfmt 3 (the skill's text-layout example) and checks each query's V-(D)-J
# rearrangement summary names the same top V, D and J genes and calls the chain productive.
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
            ev[f[1]] = f[2]; ed[f[1]] = f[3]; ej[f[1]] = f[4]; ei[f[1]] = f[6]
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
            diff = (f[col["v_identity"]] + 0) - ei[id]
            if (diff < 0) diff = -diff
            if (diff > 0.002) { printf "%s %s: v_identity %s, expected %s\n", group, id, f[col["v_identity"]], ei[id]; ok = 0 }
            if (ok) got[id] = 1; else bad++
        }
        for (id in want) if (!(id in got)) { printf "%s: %s missing from the AIRR output\n", group, id; bad++ }
        printf "%s: %d sequences, %d failures\n", group, rows - 1, bad
        exit bad ? 1 : 0
    }'
done

# -outfmt 3, IgBLAST's own text layout (the skill's second command): each query's block opens
# with its V-(D)-J rearrangement summary, then junction and sub-region details, then the
# per-region alignment summary, and ends in BLAST-like pairwise alignments.
igblastn \
    -germline_db_V "$GL/imgt_human_ig_v.fa" \
    -germline_db_D "$GL/imgt_human_ig_d.fa" \
    -germline_db_J "$GL/imgt_human_ig_j.fa" \
    -organism human -ig_seqtype Ig \
    -auxiliary_data /usr/local/share/igblast/optional_file/human_gl.aux \
    -query fixture/heavy.fa -out "$work/heavy_fmt3.txt" -outfmt 3 -num_threads 2

awk -v tsv="$work/heavy_fmt3.txt" 'BEGIN {
    n = 0
    while ((getline line < "fixture/expected.tsv") > 0) {
        if (n++ == 0) continue
        split(line, f, "\t")
        if (substr(f[1], 1, 1) != "H") continue
        ev[f[1]] = f[2]; ed[f[1]] = f[3]; ej[f[1]] = f[4]
        want[f[1]] = 1
    }
    bad = 0; queries = 0; summaries = 0; alignments = 0
    id = ""
    while ((getline line < tsv) > 0) {
        if (line ~ /^Query= /) {
            split(line, q, " ")
            id = q[2]
            if (id in want) queries++
            want_summary = 0
        } else if (line ~ /^V-\(D\)-J rearrangement summary for query sequence/) {
            want_summary = 1
        } else if (line ~ /^Alignment summary between query and top germline V gene hit/) {
            alignments++
        } else if (want_summary) {
            want_summary = 0
            summaries++
            split(line, f, "\t")
            v_ok = 0; m = split(f[1], vs, ",")
            for (i = 1; i <= m; i++) if (vs[i] == ev[id]) v_ok = 1
            d_ok = 0; m = split(f[2], ds, ",")
            for (i = 1; i <= m; i++) if (ds[i] == ed[id]) d_ok = 1
            j_ok = 0; m = split(f[3], js, ",")
            for (i = 1; i <= m; i++) if (js[i] == ej[id]) j_ok = 1
            if (!v_ok) { printf "fmt3 %s: top V %s, expected %s\n", id, f[1], ev[id]; bad++ }
            if (!d_ok) { printf "fmt3 %s: top D %s, expected %s\n", id, f[2], ed[id]; bad++ }
            if (!j_ok) { printf "fmt3 %s: top J %s, expected %s\n", id, f[3], ej[id]; bad++ }
            if (f[7] != "Yes") { printf "fmt3 %s: productive %s, expected Yes\n", id, f[7]; bad++ }
        }
    }
    if (queries != 6) { printf "fmt3: %d query blocks, expected 6\n", queries; bad++ }
    if (summaries != 6) { printf "fmt3: %d rearrangement summaries, expected 6\n", summaries; bad++ }
    if (alignments != 6) { printf "fmt3: %d alignment summaries, expected 6\n", alignments; bad++ }
    printf "heavy -outfmt 3: %d queries, %d rearrangement summaries, %d failures\n", queries, summaries, bad
    exit bad ? 1 : 0
}'

echo "igblast test passed"

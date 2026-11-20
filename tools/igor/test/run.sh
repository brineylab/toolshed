set -euo pipefail
w=$(mktemp -d)
igor_() { igor -set_wd "$w" -threads 2 "$@" > "$w/log" 2>&1 || { tail -30 "$w/log"; exit 1; }; }

# Generate, read back as FASTA (with a blank line, which 1.4.0 read past), align, and score against the same model.
# With seed 7, every sequence a model generates has Pgen > 0 under it, and its best scenario names the V gene it was
# generated from: 200 of 200 TRB, 40 of 40 IGH. Two threads take IGoR's OpenMP path.
check() {  # batch chain n
  local b=$1 c=$2 n=$3
  igor_ -batch "$b" -species human -chain "$c" -generate "$n" --noerr --CDR3 --seed 7
  awk -F';' 'FNR > 1 { print ">" $1; print $2; if (FNR == 3) print "" }' "$w/${b}_generated/generated_seqs_noerr.csv" > "$w/$b.fasta"
  igor_ -batch "$b" -species human -chain "$c" -read_seqs "$w/$b.fasta" -align --all
  igor_ -batch "$b" -species human -chain "$c" -evaluate -output --Pgen --scenarios 1
  got=$(awk -F';' 'FNR == 1 { next }
    FILENAME ~ /realizations/ { v[$1] = $2; next }
    FILENAME ~ /Pgen/ { if ($2 + 0 > 0) p++; next }
    v[$1] == $4 { s++ }
    END { print p + 0, s + 0 }' \
    "$w/${b}_generated/generated_realizations_noerr.csv" "$w/${b}_output/Pgen_counts.csv" "$w/${b}_output/best_scenarios_counts.csv")
  read -r p v <<< "$got"
  echo "$b: $p of $n with Pgen > 0, $v of $n with the generating V gene as best scenario"
  [[ $p -eq $n && $v -eq $n ]]
  [[ $(awk -F, 'NR > 1' "$w/${b}_generated/generated_seqs_noerr_CDR3_info.csv" | wc -l) -eq $n ]]
}
check trb beta 200
check igh heavy_naive 40

# Infer a TRB model from those reads, then generate from it.
igor_ -batch trb -species human -chain beta -infer --N_iter 2
m="$w/trb_inference"
[[ -s "$m/final_parms.txt" && -s "$m/final_marginals.txt" ]]
igor_ -batch trb -set_custom_model "$m/final_parms.txt" "$m/final_marginals.txt" -evaluate -output --Pgen
[[ $(awk -F';' 'NR > 1 && $2 + 0 > 0' "$w/trb_output/Pgen_counts.csv" | wc -l) -eq 200 ]]
igor_ -batch fromtrb -set_custom_model "$m/final_parms.txt" "$m/final_marginals.txt" -generate 10 --noerr --seed 7
[[ $(awk 'NR > 1' "$w/fromtrb_generated/generated_seqs_noerr.csv" | wc -l) -eq 10 ]]
echo "inferred a TRB model, scored the 200 reads under it, and generated 10 sequences from it"

# Every chain the image ships a model for generates.
for c in alpha kappa lambda heavy_memory; do igor_ -batch "$c" -species human -chain "$c" -generate 5 --seed 7; done
igor_ -batch mouse -species mouse -chain beta -generate 5 --seed 7
[[ -s /usr/local/share/igor/source/COPYING ]]
echo ok

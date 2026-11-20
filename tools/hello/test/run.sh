set -eu
out=$(hello)
echo "$out"
case "$out" in *"os="*"user="*"gpu="*"hostname="*"cpus="*) ;; *) exit 1 ;; esac
host=$(printf '%s\n' "$out" | sed -n 's/^hostname=//p')
[ -n "$host" ]
cpus=$(printf '%s\n' "$out" | sed -n 's/^cpus=//p')
case "$cpus" in ''|*[!0-9]*) exit 1 ;; esac
[ "$cpus" -ge 1 ]

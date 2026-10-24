---
name: hello
description: "Checks that this machine or cluster runs toolshed images: prints the OS, the user, and the GPU it sees. Reach for it first on a new cluster, before any real tool."
metadata:
  version: 1.0.0
---
# hello

Image: `ghcr.io/brineylab/toolshed/hello:1.0.0`

## Commands

`hello` prints three lines and exits 0:

```
os=<the image's OS id, alpine>
user=<the uid the container runs as>
gpu=<the first GPU's name from nvidia-smi, or none>
```

## Inputs and outputs

None. It reads nothing and writes only to stdout.

## Compute

No GPU needed; with one, it names it. Seconds.

## Running it

On a laptop:

```bash
podman run --rm ghcr.io/brineylab/toolshed/hello:1.0.0 hello
```

On a Slurm cluster with pyxis:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#brineylab/toolshed/hello:1.0.0 hello
```

## Gotchas

- Under pyxis the user is the node user (`user=` your uid), not root as under podman.
- `gpu=none` on a GPU node means the job did not request one (`--gres=gpu:1`) or the node's driver is not visible in the container.

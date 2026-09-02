# harbor-singularity-hpc

A drop-in **writable-rootfs Singularity/Apptainer environment** for
[harbor](https://github.com/NVIDIA/harbor), so agent benchmarks like SWE-bench
run **on the compute node** (PBS/SLURM) with no Docker, no Modal, and no
site-admin changes — on clusters whose Singularity can't produce a writable
container rootfs by default.

It is a thin **subclass** of harbor's stock `SingularityEnvironment`, selected
by import path. **It patches no harbor source**, so it installs and upgrades
like any other dependency.

## Why

On many HPC nodes `/etc/fuse.conf` lacks `user_allow_other`. Under `--fakeroot`
that stops squashfuse from FUSE-mounting a `.sif`, and Singularity falls back to
extraction + *underlay* — which has no overlayfs, so both `--writable-tmpfs` and
`--overlay` silently degrade to a **read-only** rootfs. harbor's in-container
server then can't create its venv and dies with *"Server failed to start on
port …"*. This package sidesteps that by extracting the image into a per-session
**directory** and running `--fakeroot --writable <dir>` (a plain directory needs
neither a squashfuse mount nor overlayfs).

Along the way it folds in the other fixes needed to run SWE-bench under
Singularity at scale:

| Behaviour | What it does |
|---|---|
| **Dockerfile `FROM` fallback** | SWE-bench tasks set no `[environment].docker_image`; they declare the image via `FROM …` in `environment/Dockerfile`, which harbor's Singularity backend never builds. We parse that `FROM` and pull it. |
| **Writable sandbox rootfs** | `--fakeroot --writable <dir>` instead of `--writable-tmpfs <sif>`; bind destinations and `--pwd` are pre-created in the sandbox (a `--writable` rootfs can't auto-create mountpoints). |
| **Node-local, resume-safe cache** | Default image cache resolves **live** to `$PBS_LOCALDIR` / `$SLURM_TMPDIR`, so a path is never baked stale into a chunked-resume job config. |
| **Pull retry + semaphore** | Bounded retries around `singularity pull` and a process-wide pull throttle, to ride out Docker Hub's *"unexpected end of JSON input"* under concurrency. |
| **Long exec HTTP timeout** | mini-swe-agent runs its whole loop as one `exec`; harbor's 600s client cap kills it. The floor is raised (default 86400s) so the trial's own budget governs. |

## Install

```bash
pip install harbor-singularity-hpc     # into the same env as harbor
# or, from a checkout:
pip install -e .
```

harbor is a **peer dependency** — it must already be installed. The pinned range
(`harbor>=0.22,<0.23`) is the *tested* one; see [Compatibility](#compatibility).

## Use

Select the environment by import path (works on any harbor build):

```bash
harbor run <dataset> ... \
  --environment-import-path harbor_singularity_hpc.environment:SingularityWritableEnvironment \
  --environment-kwarg singularity_writable_sandbox=true
```

The writable sandbox is **on by default** for this class; pass
`--environment-kwarg singularity_writable_sandbox=false` to fall back to stock
`--writable-tmpfs` behaviour (e.g. on a node where FUSE *is* configured).

### Environment variables

| Var | Default | Meaning |
|---|---|---|
| `HB_SINGULARITY_MAX_PULLS` | `4` | Max concurrent `singularity pull`s in one process. |
| `HB_SINGULARITY_HTTP_TIMEOUT` | `86400` | Exec HTTP-client timeout floor, seconds. |

### Constructor kwargs (`--environment-kwarg`)

| Kwarg | Default | Meaning |
|---|---|---|
| `singularity_writable_sandbox` | `true` | Use the writable sandbox rootfs. |
| `singularity_sandbox_dir` | node-local scratch | Where per-session sandboxes are built. |
| `singularity_image_cache_dir` | node-local, live | Set a **shared-FS** path to persist the sif cache across resume chunks. |

Plus every kwarg harbor's `SingularityEnvironment` already accepts
(`singularity_force_pull`, `singularity_no_mount`, …). Keep bind-paths mounted
(`singularity_no_mount=home,tmp`) so `/etc/resolv.conf` reaches the container and
in-container pip has DNS.

## How it stays upgrade-safe

The overrides deliberately depend only on harbor's stable **contracts**, not on
the bodies of its large methods:

- The writable-rootfs swap is done by rewriting the `singularity exec` **argv**
  (a small `asyncio.create_subprocess_exec` wrapper keyed on a per-session sif→
  sandbox registry) rather than re-implementing harbor's ~350-line
  `_start_server`.
- The exec timeout is raised by wrapping harbor's httpx client, not by copying
  its 80-line `exec`.
- Pull retry wraps `super()._convert_docker_to_sif(...)`.

`tests/test_overrides.py` includes **drift guards** that fail if a harbor upgrade
moves any of these seams (a renamed method, a changed signature, a different
exec argv, a changed default timeout). When one fails, re-vet the override and
widen the pin in `pyproject.toml`.

```bash
pip install -e '.[test]' && pytest
```

## Compatibility

Tested against **harbor 0.22.x**. Newer harbor may work unchanged — run the
tests first. The one process-global side effect is a guarded wrapper around
`asyncio.create_subprocess_exec`, installed on first construction; it is a strict
no-op for any subprocess that is not one of this package's writable-sandbox
`singularity exec` launches.

## The proper fix

Site-side: add `user_allow_other` to `/etc/fuse.conf`. That restores
Singularity's faster RAM-overlay path and makes the writable-sandbox mode
unnecessary. The `FROM` fallback and pull retry are worth upstreaming to harbor
regardless of FUSE config.

## License

Apache-2.0, matching harbor.

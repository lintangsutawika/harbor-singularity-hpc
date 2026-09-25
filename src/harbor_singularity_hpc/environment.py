"""A writable-rootfs Singularity environment for FUSE-restricted HPC clusters.

``SingularityWritableEnvironment`` subclasses harbor's stock
``SingularityEnvironment`` and adds, as small overrides, everything that
previously lived as source patches against harbor:

* **Dockerfile dep-layer baking** -- when a task's ``environment/Dockerfile`` has
  ``RUN``/``COPY``/``ENV`` layers, bake them onto the base sif as a chain of derived
  sifs (node-local, cached per layer by base+instructions+COPY-source contents), so
  singularity honors the Dockerfile the way Modal/Docker do -- not just its ``FROM``.
  A failed bake fails the trial, as a failed image build would.
* **Dockerfile ``FROM`` fallback** -- run SWE-bench (and other Dockerfile-defined)
  tasks that set no ``[environment].docker_image`` in ``task.toml``.
* **Writable sandbox rootfs** -- extract the image to a per-session *directory*
  and run ``--fakeroot --writable <dir>`` instead of ``--writable-tmpfs <sif>``,
  for nodes whose ``/etc/fuse.conf`` lacks ``user_allow_other`` (there both
  ``--writable-tmpfs`` and ``--overlay`` silently degrade to a read-only rootfs).
* **Host ``/dev`` + resolver rebind** -- a plain ``--writable`` dir has no tmpfs
  ``/dev`` and an empty ``/etc/resolv.conf``/``/etc/hosts``; the rewriter binds the
  host copies back in so ``/dev/null`` exists (bootstrap/apt/server) and in-sandbox
  ``pip``/``uv`` can resolve DNS. Idempotent and skipped for already-bound paths.
* **Tag+digest ref normalization** -- ``singularity pull`` rejects a Docker ref carrying
  both a tag and a digest ("...currently not supported"); strip the tag and keep the digest
  so harbor's pinned task images (``repo:tag@sha256:...``) pull.
* **Node-local, resume-safe image cache** -- default to ``$PBS_LOCALDIR`` /
  ``$SLURM_TMPDIR`` resolved live, so a cache path is never baked stale into a
  chunked-resume job config.
* **Docker Hub pull retry + a process-wide pull semaphore** -- survive the
  transient "unexpected end of JSON input" / "conveyor failed to get" failures
  that OCI pulls hit under concurrency.
* **Long exec HTTP timeout** -- mini-swe-agent runs its whole loop as a single
  ``exec`` with no ``timeout_sec``; harbor's 600s client cap kills it well under
  the trial's real budget. Raise the floor so the outer budget governs.

Nothing here edits harbor's source. Select it with harbor's import-path route::

    harbor run ... \\
      --environment-import-path harbor_singularity_hpc.environment:SingularityWritableEnvironment \\
      --environment-kwarg singularity_writable_sandbox=true

The overrides depend only on harbor's stable *contracts* -- method signatures
and the ``singularity exec`` argv shape -- not on the bodies of its large
methods, so a harbor point release is unlikely to break them. ``tests/`` guards
the seams that would; pin the tested harbor range in ``pyproject.toml``.

The proper fix is site-side (add ``user_allow_other`` to ``/etc/fuse.conf``),
which restores singularity's faster RAM-overlay path and makes the writable
sandbox unnecessary; the ``FROM`` fallback and pull retry are worth upstreaming
to harbor regardless.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from harbor.environments.singularity.singularity import SingularityEnvironment

# Harbor's hardcoded per-request HTTP timeout on the exec path (seconds). We only
# raise a request that used exactly this default, leaving an explicit
# ``timeout_sec`` request untouched -- matching the original patch semantics.
_HARBOR_DEFAULT_HTTP_TIMEOUT = 600

# How many times to retry a flaky ``singularity pull`` before giving up.
_MAX_PULL_ATTEMPTS = 5


def _install_argv_rewrite() -> None:
    """Idempotently wrap ``asyncio.create_subprocess_exec`` so a
    ``singularity exec --writable-tmpfs <sif> ...`` launched by harbor's stock
    ``_start_server`` is rewritten to ``--writable <sandbox>`` whenever ``<sif>``
    is a sandbox registered in ``SingularityWritableEnvironment._WRITABLE_REGISTRY``.

    Keyed by the sif path carried in the argv (unique per environment), so it is
    safe under many concurrent trials in one process: each exec is rewritten
    using *its own* sandbox, regardless of interleaving. It is inert for every
    other subprocess -- ``singularity build``/``pull`` and any argv whose image
    is not a registered writable sandbox pass straight through. This is the one
    place we touch a process-global; the registry guard keeps it a no-op for all
    code that is not one of our writable-sandbox launches.
    """
    current = getattr(asyncio, "create_subprocess_exec", None)
    if current is None or getattr(current, "_hb_writable_wrapped", False):
        return
    _orig = current

    async def _wrapped(*args: Any, **kwargs: Any):
        return await _orig(*_rewrite_singularity_argv(list(args)), **kwargs)

    _wrapped._hb_writable_wrapped = True  # type: ignore[attr-defined]
    _wrapped._hb_orig = _orig  # type: ignore[attr-defined]
    asyncio.create_subprocess_exec = _wrapped  # type: ignore[assignment]


def _rewrite_singularity_argv(argv: list) -> list:
    """Rewrite a ``singularity exec --writable-tmpfs <registered-sif>`` argv to
    use ``--writable <sandbox>``, pre-creating every ``-B`` destination and the
    ``--pwd`` workdir inside the sandbox first (``--writable`` cannot auto-create
    bind mountpoints). Any other argv is returned unchanged."""
    if (
        len(argv) < 3
        or os.path.basename(str(argv[0])) != "singularity"
        or str(argv[1]) != "exec"
        or "--writable-tmpfs" not in argv
    ):
        return argv

    registry = SingularityWritableEnvironment._WRITABLE_REGISTRY
    sandbox: Path | None = None
    image_index: int | None = None
    for i, token in enumerate(argv):
        if str(token) in registry:
            sandbox = registry[str(token)]
            image_index = i
            break
    if sandbox is None or image_index is None:
        return argv

    # Pre-create the bind destinations and workdir inside the sandbox.
    dests: list[str] = []
    for i, token in enumerate(argv):
        if token == "-B" and i + 1 < len(argv):
            spec = str(argv[i + 1])
            dst = spec.split(":", 1)[1] if ":" in spec else spec
            dests.append(dst.split(":", 1)[0])  # drop any :ro/:rw option
        elif token == "--pwd" and i + 1 < len(argv):
            dests.append(str(argv[i + 1]))
    for dst in dests:
        if dst.startswith("/"):
            (Path(sandbox) / dst.lstrip("/")).mkdir(parents=True, exist_ok=True)

    new = list(argv)
    new[new.index("--writable-tmpfs")] = "--writable"
    new[image_index] = str(sandbox)

    # A plain ``--writable`` sandbox dir (unlike ``--writable-tmpfs``) has no tmpfs
    # ``/dev`` and carries the image's own (empty) ``/etc/resolv.conf`` +
    # ``/etc/hosts``. Under ``--containall`` that means no ``/dev/null`` -- which
    # breaks bootstrap redirects, apt, and the in-container server -- and DNS
    # failures ("Temporary failure in name resolution"), so in-sandbox ``pip``/``uv``
    # cannot reach PyPI. The network namespace is shared (no ``--net``), so binding
    # the host copies back in restores both. Idempotent: skip a path already bound
    # (harbor's argv, or a repeat call) or absent on the host.
    for hostpath in ("/dev", "/etc/resolv.conf", "/etc/hosts"):
        already = any(
            str(new[i]) in ("-B", "--bind") and i + 1 < len(new)
            and str(new[i + 1]).split(":", 1)[0] == hostpath
            for i in range(len(new))
        )
        if not already and os.path.exists(hostpath):
            new[2:2] = ["--bind", hostpath]
    return new


def _singularity_safe_ref(ref: str) -> str:
    """Make a Docker image reference acceptable to ``singularity pull``.

    harbor hands the task image as ``repo:tag@sha256:<digest>`` (both a tag AND a digest).
    Docker accepts that, but apptainer/singularity rejects it -- "Docker references with
    both a tag and digest are currently not supported". When both are present, drop the tag
    and keep the digest: the digest is the exact content pin, so the tag is redundant. Refs
    with only a tag, or only a digest, pass through unchanged. A registry ``host:port`` in the
    name is preserved (only the tag in the final path component is stripped)."""
    if not ref or "@" not in ref:
        return ref
    name_tag, _, digest = ref.partition("@")
    last = name_tag.rsplit("/", 1)[-1]  # final path component may carry "repo:tag"
    if ":" in last:                     # strip the tag, keep any registry host:port prefix
        name_tag = name_tag[: len(name_tag) - len(last)] + last.rsplit(":", 1)[0]
    return f"{name_tag}@{digest}"


# Where COPY sources are staged inside the image during a bake. ``%files`` targets
# under ``/tmp`` (the usual Dockerfile staging dir) are invisible to ``%post``:
# singularity binds a host ``/tmp`` over the image's during the build, so
# ``COPY x.sh /tmp/x.sh`` + ``RUN bash /tmp/x.sh`` dies with "No such file". Staging
# every source here and replaying the COPY inside ``%post`` sidesteps the mask.
_BAKE_CTX = "/.harbor-ctx"


def _parse_dockerfile(dockerfile: Path) -> list[tuple[str, Any]]:
    """Parse a Dockerfile into ``(instruction, arg)`` steps, in order.

    Folds ``\\`` line continuations and drops comments. Only the instructions
    that shape the filesystem or build env are kept: ``RUN`` (arg: shell text),
    ``COPY``/``ADD`` (arg: ``(srcs, dst)``), ``ENV``/``ARG`` (arg: the raw body)
    and ``WORKDIR`` (arg: path). ``FROM``/``USER``/``CMD``/... are dropped.
    """
    if not dockerfile.exists():
        return []
    try:
        lines = dockerfile.read_text().splitlines()
    except OSError:
        return []

    logical: list[str] = []
    buf: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not buf and (not line or line.startswith("#")):
            continue
        if buf and line.startswith("#"):       # comments inside a continuation
            continue
        if line.endswith("\\"):
            buf.append(line[:-1].strip())
            continue
        buf.append(line)
        logical.append(" ".join(b for b in buf if b))
        buf = []
    if buf:
        logical.append(" ".join(b for b in buf if b))

    steps: list[tuple[str, Any]] = []
    for line in logical:
        head, _, body = line.partition(" ")
        op = head.upper()
        body = body.strip()
        if op == "RUN":
            if body.startswith("["):           # exec form: RUN ["bash", "-c", "..."]
                try:
                    body = shlex.join(json.loads(body))
                except ValueError:
                    pass
            steps.append(("RUN", body))
        elif op in ("COPY", "ADD"):
            parts = [p for p in shlex.split(body) if not p.startswith("--")]
            if any(p.startswith("--from") for p in shlex.split(body)):
                raise ValueError(f"multi-stage COPY is not supported: {line}")
            if len(parts) >= 2:
                steps.append(("COPY", (parts[:-1], parts[-1])))
        elif op in ("ENV", "ARG"):
            if body:
                steps.append((op, body))
        elif op == "WORKDIR":
            steps.append(("WORKDIR", body))
    return steps


def _env_export(op: str, body: str) -> str:
    """``ENV K=V ...`` / legacy ``ENV K V`` / ``ARG K[=V]`` -> a shell ``export``."""
    if op == "ARG" and "=" not in body:
        return ""                              # bare ARG: nothing to set
    if "=" in body.split(None, 1)[0]:
        return f"export {body}"
    key, _, value = body.partition(" ")
    return f'export {key}="{value.strip()}"'


def _dockerfile_layers(dockerfile: Path, build_context: Path | None = None) -> list[tuple[str, str]]:
    """Translate a task Dockerfile into a chain of Singularity recipe fragments.

    Returns ``[(cache_material, fragment), ...]`` -- one entry per layer, each to be
    built with ``Bootstrap: localimage`` on top of the previous layer's sif -- or
    ``[]`` when the Dockerfile has no ``RUN``/``COPY``/``ENV`` (serve the base sif).

    A new layer starts at every ``COPY`` that follows a ``RUN``, so a shared prefix
    (e.g. ``COPY base_install.sh`` + ``RUN bash base_install.sh``) builds once and is
    reused by every task whose prefix is byte-identical -- Docker's layer cache, at
    COPY granularity. ``cache_material`` covers the instructions *and* the contents
    of every COPY source up to and including that layer.

    Semantics follow ``docker build``: each ``RUN`` runs in its own subshell from the
    current ``WORKDIR`` with every prior ``ENV``/``ARG`` exported; ``COPY`` sources
    resolve against ``build_context`` (default: the Dockerfile's dir); ``ENV`` also
    lands in the ``%environment`` of the layer that sets it so the agent and
    verifier see it at runtime.
    """
    steps = _parse_dockerfile(dockerfile)
    if not any(op in ("RUN", "COPY", "ENV") for op, _ in steps):
        return []
    ctx = (build_context or dockerfile.parent).resolve()

    # Group into layers: close the current layer when a COPY follows a RUN.
    groups: list[list[tuple[str, Any]]] = [[]]
    seen_run = False
    for op, arg in steps:
        if op == "COPY" and seen_run:
            groups.append([])
            seen_run = False
        groups[-1].append((op, arg))
        seen_run = seen_run or op == "RUN"

    layers: list[tuple[str, str]] = []
    material = hashlib.sha256()
    envs: list[tuple[str, str]] = []           # (op, body), cumulative across layers
    workdir = "/"
    for group in groups:
        files: list[str] = []
        ops: list[str] = []
        # %post already sources the base image's %environment (earlier layers'
        # ENV); only build-time ARGs need re-exporting.
        prelude = [e for e in (_env_export(o, b) for o, b in envs if o == "ARG") if e]
        start_workdir = workdir
        for op, arg in group:
            material.update(f"{op}\0{arg!r}\n".encode())
            if op in ("ENV", "ARG"):
                envs.append((op, arg))
                ops.append(_env_export(op, arg))
            elif op == "WORKDIR":
                workdir = arg if arg.startswith("/") else f"{workdir.rstrip('/')}/{arg}"
                ops.append(f"mkdir -p {shlex.quote(workdir)} && cd {shlex.quote(workdir)}")
            elif op == "RUN":
                ops.append(f"( cd {shlex.quote(workdir)} && {arg} )")
            elif op == "COPY":
                srcs, dst = arg
                matches: list[Path] = []
                for src in srcs:
                    hits = sorted(ctx.glob(src)) if any(c in src for c in "*?[") else [ctx / src]
                    matches.extend(h.resolve() for h in hits)
                into_dir = dst.endswith("/") or len(matches) > 1
                qdst = shlex.quote(dst.rstrip("/") or "/")
                for src_path in matches:
                    if not src_path.exists():
                        raise FileNotFoundError(f"COPY source not in build context: {src_path}")
                    material.update(_path_digest(src_path).encode())
                    staged = f"{_BAKE_CTX}/{len(files)}"
                    files.append(f"    {src_path} {staged}")
                    if src_path.is_dir():           # docker copies a dir's *contents*
                        ops.append(f"mkdir -p {qdst} && cp -a {staged}/. {qdst}/")
                    elif into_dir:
                        ops.append(f"mkdir -p {qdst} && cp -a {staged} {qdst}/{shlex.quote(src_path.name)}")
                    else:
                        ops.append(
                            f"if [ -d {qdst} ]; then cp -a {staged} {qdst}/{shlex.quote(src_path.name)}; "
                            f'else mkdir -p "$(dirname {qdst})" && cp -a {staged} {qdst}; fi'
                        )
        # singularity *appends* each layer's %environment to the base image's, so
        # emit only the ENVs this layer introduces (cumulative would re-apply e.g.
        # PATH=/x:$PATH once per layer).
        env_lines = [_env_export(op, arg) for op, arg in group if op == "ENV"]

        block: list[str] = []
        if files:
            block.append("%files")
            block.extend(files)
        block.append("%post")
        block.append("    set -e")
        block.extend(f"    {p}" for p in prelude)
        block.append(f"    mkdir -p {shlex.quote(start_workdir)} && cd {shlex.quote(start_workdir)}")
        block.extend(f"    {o}" for o in ops if o)
        if files:
            block.append(f"    rm -rf {_BAKE_CTX}")
        if env_lines:
            block.append("%environment")
            block.extend(f"    {e}" for e in env_lines)
        layers.append((material.hexdigest(), "\n".join(block) + "\n"))
    return layers


def _path_digest(path: Path) -> str:
    """Content digest of a COPY source (file, or every file under a dir)."""
    h = hashlib.sha256()
    paths = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for p in paths:
        h.update(str(p.relative_to(path) if path.is_dir() else p.name).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


class _TimeoutFloorClient:
    """Transparent proxy over harbor's httpx exec client that raises a request
    whose timeout is exactly harbor's default (i.e. the no-``timeout_sec`` exec
    path) to ``floor`` seconds. Explicit shorter timeouts are left alone."""

    def __init__(self, inner: Any, floor: float) -> None:
        self._inner = inner
        self._floor = floor

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def post(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("timeout") == _HARBOR_DEFAULT_HTTP_TIMEOUT and self._floor > _HARBOR_DEFAULT_HTTP_TIMEOUT:
            kwargs["timeout"] = self._floor
        return await self._inner.post(*args, **kwargs)

    async def aclose(self) -> Any:
        return await self._inner.aclose()


def _descendant_pids(root_pid: int) -> list[int]:
    """Return all descendants of root_pid, deepest descendants first."""
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    children: dict[int, list[int]] = {}

    for line in result.stdout.splitlines():
        try:
            pid_s, ppid_s = line.split()
            pid = int(pid_s)
            ppid = int(ppid_s)
        except (ValueError, TypeError):
            continue

        children.setdefault(ppid, []).append(pid)

    descendants: list[int] = []

    def walk(pid: int) -> None:
        for child in children.get(pid, []):
            walk(child)
            descendants.append(child)

    walk(root_pid)
    return descendants

class SingularityWritableEnvironment(SingularityEnvironment):
    """``SingularityEnvironment`` that yields a writable rootfs on FUSE-restricted
    HPC nodes, runs Dockerfile-defined tasks, and hardens image pulls. See the
    module docstring for the full list and the ``--environment-import-path``
    invocation."""

    # Package version marker. Bump when the override behaviour changes.
    _HB_HPC_PATCHSET = "4"

    # Process-wide throttle on concurrent ``singularity pull``s. Many at once race
    # the shared OCI blob cache and burst Docker Hub's rate limit. Agent
    # concurrency is unaffected -- only pulls are limited. Tune via
    # ``HB_SINGULARITY_MAX_PULLS``.
    _PULL_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("HB_SINGULARITY_MAX_PULLS", "4")))

    # sif path (str) -> writable sandbox dir, for every live writable environment
    # in this process. Consulted by ``_rewrite_singularity_argv``.
    _WRITABLE_REGISTRY: dict[str, Path] = {}

    def __init__(
        self,
        *args: Any,
        singularity_image_cache_dir: Path | str | None = None,
        singularity_writable_sandbox: bool = True,
        singularity_sandbox_dir: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            singularity_image_cache_dir=singularity_image_cache_dir,
            **kwargs,
        )
        # When no cache dir was given, prefer node-local scratch resolved LIVE
        # (harbor's default is a throwaway mkdtemp that re-pulls every trial).
        if not singularity_image_cache_dir:
            self._image_cache_dir = self._default_image_cache_dir()

        self._writable_sandbox = singularity_writable_sandbox
        self._sandbox_dir_root = singularity_sandbox_dir
        self._sandbox_path: Path | None = None
        self._http_timeout_floor = int(
            os.environ.get("HB_SINGULARITY_HTTP_TIMEOUT", "86400")
        )
        _install_argv_rewrite()

    # -- Dockerfile FROM fallback -----------------------------------------

    @property
    def _docker_image(self) -> str | None:
        # Stock harbor returns only task.toml [environment].docker_image.
        # SWE-bench tasks declare their image via `FROM <image>` in
        # environment/Dockerfile, which the Singularity backend never builds --
        # fall back to that FROM so those tasks run. bootstrap.sh installs at
        # runtime the deps the skipped RUN layers would have added.
        if self.task_env_config.docker_image:
            return self.task_env_config.docker_image
        return self._docker_image_from_dockerfile()

    def _docker_image_from_dockerfile(self) -> str | None:
        """Return the effective (last, non-comment) ``FROM`` image, or ``None``."""
        if not self._dockerfile_path.exists():
            return None
        image: str | None = None
        try:
            for raw in self._dockerfile_path.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.upper().startswith("FROM "):
                    ref = line.split(None, 1)[1].strip()
                    upper = ref.upper()
                    if " AS " in upper:  # drop a trailing "AS <stage>"
                        ref = ref[: upper.index(" AS ")].strip()
                    image = ref
        except Exception:
            return None
        return image

    # -- cache dir + sandbox helpers --------------------------------------

    def _default_image_cache_dir(self) -> Path:
        """Node-local ($PBS_LOCALDIR / $SLURM_TMPDIR), resolved LIVE per process
        so it is never a stale absolute path baked into a resumed job config.
        Shared by all trials in the job. Falls back to a private temp dir."""
        for var in ("PBS_LOCALDIR", "SLURM_TMPDIR", "TMPDIR"):
            val = os.environ.get(var)
            if val:
                return Path(val) / "harbor_sif_cache"
        return Path(tempfile.mkdtemp(prefix="singularity_cache_"))

    def _sandbox_root(self) -> Path:
        """Node-local dir under which per-session sandboxes are built."""
        if self._sandbox_dir_root:
            return Path(self._sandbox_dir_root)
        for var in ("PBS_LOCALDIR", "SLURM_TMPDIR", "TMPDIR"):
            val = os.environ.get(var)
            if val:
                return Path(val)
        return Path(tempfile.gettempdir())

    async def _build_writable_sandbox(self, sif_path: Path) -> Path:
        """Extract ``sif_path`` into a per-session writable sandbox directory.

        ``--fix-perms`` gives the owner rwX on every entry so the tree is
        removable later. ``singularity build --sandbox`` extracts (no FUSE
        mount), so it is unaffected by a missing ``user_allow_other``."""
        root = self._sandbox_root()
        root.mkdir(parents=True, exist_ok=True)
        sandbox = root / f"hbsbx_{self.session_id}"
        await self._remove_sandbox(sandbox)
        cmd = ["singularity", "build", "--fix-perms", "--sandbox", str(sandbox), str(sif_path)]
        self.logger.debug(f"Building writable sandbox: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"Failed to build writable sandbox from {sif_path}: "
                f"{stderr.decode(errors='replace')}"
            )
        return sandbox

    async def _remove_sandbox(self, sandbox: Path | None) -> None:
        """Best-effort removal of a sandbox dir. Node-local scratch is job-scoped
        and reclaimed at job end, so leftovers are not fatal."""
        if not sandbox or not sandbox.exists():
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                "chmod", "-R", "u+rwX", str(sandbox),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception:
            pass
        shutil.rmtree(sandbox, ignore_errors=True)

    # -- pull retry + semaphore -------------------------------------------

    async def _convert_docker_to_sif(self, docker_image: str, *, force_pull: bool = False) -> Path:
        """Wrap harbor's converter with a process-wide pull semaphore and bounded
        retries. Each retry re-runs the base method, which returns immediately if
        the image is already cached, so a mid-run success costs nothing extra."""
        # singularity pull rejects a tag+digest ref; normalize to digest-only.
        docker_image = _singularity_safe_ref(docker_image)
        async with type(self)._PULL_SEMAPHORE:
            last_error: Exception | None = None
            for attempt in range(_MAX_PULL_ATTEMPTS):
                if attempt:
                    delay = min(5 * (2 ** (attempt - 1)), 60)
                    self.logger.warning(
                        f"singularity pull of {docker_image} failed "
                        f"(attempt {attempt}/{_MAX_PULL_ATTEMPTS}), retrying in "
                        f"{delay}s: {str(last_error)[-200:]}"
                    )
                    await asyncio.sleep(delay)
                try:
                    return await super()._convert_docker_to_sif(docker_image, force_pull=force_pull)
                except RuntimeError as e:
                    last_error = e
            raise RuntimeError(
                f"Failed to convert Docker image after {_MAX_PULL_ATTEMPTS} attempts: {last_error}"
            )


    # -- Dockerfile dep-layers -> derived sif -----------------------------

    async def _ensure_dockerfile_derived_sif(self, base_sif: Path) -> Path:
        """Bake the task's ``environment/Dockerfile`` onto ``base_sif`` and return
        the resulting sif (``base_sif`` itself if the Dockerfile has nothing to bake).

        This makes singularity honor the Dockerfile the way Modal/Docker do -- stock
        singularity only reads its ``FROM``, silently skipping the ``RUN``/``COPY``/
        ``ENV`` layers that install e.g. the verifier's pytest. Layers (see
        ``_dockerfile_layers``) are built as a chain of derived sifs in the node-local
        image cache, each keyed by the base sif + instructions + COPY-source contents,
        so a layer is built once per node and shared by every trial/task that reuses
        it. A file lock serializes concurrent builders of the same layer.

        A failed build raises (as a failed ``docker build`` / Modal image build
        would) instead of silently running the task in an unprovisioned container.
        """
        layers = _dockerfile_layers(self._dockerfile_path, self.environment_dir)
        if not layers:
            return base_sif
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)
        current = base_sif
        for n, (material, fragment) in enumerate(layers, 1):
            key = hashlib.sha256(f"{current}::{material}".encode()).hexdigest()[:16]
            derived = self._image_cache_dir / f"df_{key}.sif"
            lock_path = self._image_cache_dir / f"df_{key}.lock"
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
            try:
                await asyncio.to_thread(fcntl.flock, lock_fd, fcntl.LOCK_EX)
                if not derived.exists():
                    await self._build_derived_layer(current, fragment, derived, f"{n}/{len(layers)}")
                else:
                    self.logger.info(f"Reusing cached Dockerfile layer {n}/{len(layers)}: {derived}")
            finally:
                os.close(lock_fd)                 # releases the flock
            current = derived
        return current

    async def _build_derived_layer(self, base: Path, fragment: str, out: Path, label: str) -> None:
        def_path = out.with_suffix(".def")
        def_path.write_text(f"Bootstrap: localimage\nFrom: {base}\n\n{fragment}")
        tmp_out = out.with_name(f"{out.name}.tmp-{os.getpid()}")
        # Give %post a private /tmp: singularity otherwise binds the host's shared
        # /tmp, where concurrent builds' staged scripts would collide.
        # World-writable + sticky like a real /tmp: apt drops to its ``_apt`` user and
        # fails every fetch ("Couldn't create temporary file") under mkdtemp's 0700.
        build_tmp = Path(tempfile.mkdtemp(prefix="df_tmp_", dir=self._image_cache_dir))
        build_tmp.chmod(0o1777)
        cmd = [
            "singularity", "build", "--fakeroot", "--force",
            "--bind", f"{build_tmp}:/tmp",
            str(tmp_out), str(def_path),
        ]
        self.logger.info(f"Building Dockerfile layer {label}: {' '.join(cmd)}")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.environment_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await proc.communicate()
            log_path = out.with_suffix(".log")
            log_path.write_bytes(output)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Dockerfile layer {label} build failed ({proc.returncode}); "
                    f"recipe {def_path}, log {log_path}. Output tail:\n"
                    f"{output.decode(errors='replace')[-4000:]}"
                )
            os.replace(tmp_out, out)
        finally:
            tmp_out.unlink(missing_ok=True)
            shutil.rmtree(build_tmp, ignore_errors=True)

    # -- server launch (writable sandbox + exec timeout floor) ------------

    async def _start_server(self) -> None:
        # If the task Dockerfile adds dependency layers, bake them into a derived
        # sif and sandbox THAT, so singularity honors RUN/COPY/ENV like Modal/Docker.
        if self._sif_path is not None and self._dockerfile_path.exists():
            self._sif_path = await self._ensure_dockerfile_derived_sif(self._sif_path)
        # Build the writable sandbox from the (possibly derived) sif and register
        # it so _rewrite_singularity_argv redirects the exec harbor launches.
        # harbor's stock start() assigns self._sif_path before calling us.
        if self._writable_sandbox and self._sandbox_path is None and self._sif_path is not None:
            self._sandbox_path = await self._build_writable_sandbox(self._sif_path)
            type(self)._WRITABLE_REGISTRY[str(self._sif_path)] = self._sandbox_path

        await super()._start_server()

        # Raise the exec HTTP timeout floor once the (readiness-checked) client
        # exists, so long single-exec agent loops are governed by the outer
        # trial budget rather than harbor's 600s default.
        if self._http_client is not None and not isinstance(self._http_client, _TimeoutFloorClient):
            self._http_client = _TimeoutFloorClient(self._http_client, self._http_timeout_floor)

    async def stop(self, delete: bool) -> None:
        # Harbor's stock stop() terminates the outer Singularity process before
        # trying to kill its children. On this HPC setup, surviving Apptainer
        # descendants can be re-parented to PID 1 before Harbor's pkill runs.
        # Remember the process tree while it still belongs to this environment.
        descendant_pids: list[int] = []

        if self._server_process and self._server_process.returncode is None:
            descendant_pids = _descendant_pids(self._server_process.pid)

        try:
            await super().stop(delete)
        finally:
            # Reap only processes that belonged to this environment when
            # teardown began.
            for pid in descendant_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    pass

            if self._sandbox_path is not None:
                if self._sif_path is not None:
                    type(self)._WRITABLE_REGISTRY.pop(str(self._sif_path), None)
                await self._remove_sandbox(self._sandbox_path)
                self._sandbox_path = None

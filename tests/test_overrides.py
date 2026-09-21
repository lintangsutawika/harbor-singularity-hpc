"""Tests for harbor-singularity-hpc.

Two kinds:

* **Behaviour** -- the argv rewriter and the timeout-floor client do the right
  thing, checked without a real cluster.
* **Drift guards** -- the overrides depend on harbor contracts (method names,
  signatures, and the ``singularity exec`` argv shape emitted by harbor's stock
  ``_start_server``). These tests fail loudly if a harbor upgrade moves a seam,
  which is the signal to re-vet the overrides and widen the pin in
  ``pyproject.toml``.
"""

import asyncio
import inspect
from pathlib import Path

import pytest

from harbor.environments.singularity.singularity import SingularityEnvironment
from harbor_singularity_hpc.environment import (
    SingularityWritableEnvironment,
    _TimeoutFloorClient,
    _HARBOR_DEFAULT_HTTP_TIMEOUT,
    _rewrite_singularity_argv,
    _singularity_safe_ref,
    _dockerfile_dep_layers,
)


@pytest.mark.parametrize("ref,expected", [
    # tag + digest -> drop the tag, keep the digest
    ("harborframework/terminal-bench:layout-x-9caa9d66@sha256:c08989",
     "harborframework/terminal-bench@sha256:c08989"),
    # digest only -> unchanged
    ("repo/img@sha256:abc", "repo/img@sha256:abc"),
    # tag only -> unchanged
    ("repo/img:1.2.3", "repo/img:1.2.3"),
    # bare name -> unchanged
    ("ubuntu", "ubuntu"),
    # registry host:port preserved, only the trailing tag stripped
    ("reg.io:5000/repo/img:tag@sha256:def", "reg.io:5000/repo/img@sha256:def"),
])
def test_singularity_safe_ref(ref, expected):
    assert _singularity_safe_ref(ref) == expected


# --------------------------------------------------------------------------- #
# Drift guards: harbor contracts the overrides rely on
# --------------------------------------------------------------------------- #

def test_subclass_relationship():
    assert issubclass(SingularityWritableEnvironment, SingularityEnvironment)


@pytest.mark.parametrize(
    "name", ["_docker_image", "_dockerfile_path", "_convert_docker_to_sif",
             "_start_server", "start", "stop"],
)
def test_base_still_has_seam(name):
    assert hasattr(SingularityEnvironment, name), (
        f"harbor's SingularityEnvironment no longer exposes {name!r}; "
        "re-vet the overrides in environment.py against this harbor version."
    )


def test_convert_signature_unchanged():
    sig = inspect.signature(SingularityEnvironment._convert_docker_to_sif)
    params = list(sig.parameters)
    assert params[:2] == ["self", "docker_image"]
    assert "force_pull" in sig.parameters


def test_base_init_accepts_image_cache_dir_kwarg():
    # Our __init__ passes singularity_image_cache_dir through to super() and then
    # overrides self._image_cache_dir when it is unset.
    assert "singularity_image_cache_dir" in inspect.signature(
        SingularityEnvironment.__init__
    ).parameters


def test_start_server_still_emits_writable_tmpfs_argv():
    """Our argv rewriter keys off the exact flag/shape harbor's _start_server
    builds. If any of these tokens disappears from the source, the rewrite is a
    silent no-op and containers come up read-only -- fail here instead."""
    src = inspect.getsource(SingularityEnvironment._start_server)
    for token in ('"--writable-tmpfs"', '"singularity"', '"exec"', '"-B"',
                  '"--pwd"', "create_subprocess_exec"):
        assert token in src, (
            f"harbor's _start_server no longer contains {token}; "
            "the argv rewriter in environment.py needs updating."
        )


def test_exec_default_timeout_contract():
    """The timeout-floor client only bumps a request whose timeout equals
    harbor's hardcoded default. Two acceptable states: pristine harbor uses
    ``= 600``; an install that already raised the cap uses an
    ``HB_SINGULARITY_HTTP_TIMEOUT`` env override (in which case our floor is a
    harmless no-op, since requests already exceed 600). If neither holds, harbor
    changed the default and _HARBOR_DEFAULT_HTTP_TIMEOUT needs re-vetting."""
    src = inspect.getsource(SingularityEnvironment.exec)
    pristine = f"_DEFAULT_HTTP_TIMEOUT = {_HARBOR_DEFAULT_HTTP_TIMEOUT}" in src
    already_raised = "HB_SINGULARITY_HTTP_TIMEOUT" in src
    assert pristine or already_raised, (
        "harbor's exec() default HTTP timeout changed; update "
        "_HARBOR_DEFAULT_HTTP_TIMEOUT in environment.py."
    )


# --------------------------------------------------------------------------- #
# Behaviour: argv rewriter
# --------------------------------------------------------------------------- #

def _exec_argv(sif: str) -> list:
    """A minimal argv matching what harbor's _start_server emits."""
    return [
        "singularity", "exec",
        "--no-mount", "home",
        "--pwd", "/testbed",
        "--writable-tmpfs",
        "--fakeroot", "--containall", "--pid",
        "-B", "/tmp/staging:/staging",
        "-B", "/tmp/staging/env_files:/staging/env_files",
        sif,
        "bash", "-c", 'exec /staging/bootstrap.sh "$@"',
    ]


def test_rewrite_swaps_flag_and_image_and_creates_dests(tmp_path):
    sandbox = tmp_path / "sbx"
    sandbox.mkdir()
    sif = str(tmp_path / "img.sif")
    SingularityWritableEnvironment._WRITABLE_REGISTRY[sif] = sandbox
    try:
        out = _rewrite_singularity_argv(_exec_argv(sif))
    finally:
        SingularityWritableEnvironment._WRITABLE_REGISTRY.pop(sif, None)

    assert "--writable" in out and "--writable-tmpfs" not in out
    assert str(sandbox) in out and sif not in out
    # Every bind destination and the --pwd workdir exist inside the sandbox.
    for dst in ("testbed", "staging", "staging/env_files"):
        assert (sandbox / dst).is_dir(), f"{dst} not pre-created"


def test_rewrite_binds_dev_and_resolver(tmp_path):
    """A plain --writable dir loses tmpfs /dev + the host resolver; the rewriter
    must bind them back so /dev/null exists and in-sandbox pip/uv can resolve DNS."""
    sandbox = tmp_path / "sbx"
    sandbox.mkdir()
    sif = str(tmp_path / "img.sif")
    SingularityWritableEnvironment._WRITABLE_REGISTRY[sif] = sandbox
    try:
        out = _rewrite_singularity_argv(_exec_argv(sif))
    finally:
        SingularityWritableEnvironment._WRITABLE_REGISTRY.pop(sif, None)

    # Each host path that exists is bound exactly once via --bind <path>.
    for hostpath in ("/dev", "/etc/resolv.conf", "/etc/hosts"):
        if not Path(hostpath).exists():
            continue
        bound = [
            out[i + 1] for i in range(len(out) - 1)
            if out[i] in ("-B", "--bind") and str(out[i + 1]).split(":", 1)[0] == hostpath
        ]
        assert bound == [hostpath], f"{hostpath} should be bound exactly once, got {bound}"


def test_rewrite_dev_bind_is_idempotent(tmp_path):
    """Re-running the rewrite (or a host argv that already binds /dev) must not
    add a duplicate bind."""
    sandbox = tmp_path / "sbx"
    sandbox.mkdir()
    sif = str(tmp_path / "img.sif")
    SingularityWritableEnvironment._WRITABLE_REGISTRY[sif] = sandbox
    try:
        once = _rewrite_singularity_argv(_exec_argv(sif))
        # feed it back through unchanged-image path by re-registering the sandbox as
        # its own image key is not needed; just assert a second pass over an argv that
        # already binds /dev adds nothing new.
        argv2 = ["singularity", "exec", "--bind", "/dev", "--writable-tmpfs",
                 "-B", "/tmp/s:/staging", "--pwd", "/testbed", sif, "bash"]
        out2 = _rewrite_singularity_argv(argv2)
    finally:
        SingularityWritableEnvironment._WRITABLE_REGISTRY.pop(sif, None)
    if Path("/dev").exists():
        assert sum(1 for i in range(len(out2) - 1)
                   if out2[i] in ("-B", "--bind") and out2[i + 1] == "/dev") == 1


def test_rewrite_ignores_unregistered_sif():
    argv = _exec_argv("/tmp/not-registered.sif")
    assert _rewrite_singularity_argv(argv) is argv  # unchanged, same object


def test_rewrite_ignores_non_exec_commands(tmp_path):
    sif = str(tmp_path / "img.sif")
    SingularityWritableEnvironment._WRITABLE_REGISTRY[sif] = tmp_path / "sbx"
    try:
        build = ["singularity", "build", "--fix-perms", "--sandbox", "/x", sif]
        pull = ["singularity", "pull", sif, f"docker://busybox"]
        assert _rewrite_singularity_argv(build) is build
        assert _rewrite_singularity_argv(pull) is pull
    finally:
        SingularityWritableEnvironment._WRITABLE_REGISTRY.pop(sif, None)


# --------------------------------------------------------------------------- #
# Behaviour: timeout-floor client
# --------------------------------------------------------------------------- #

class _RecordingClient:
    def __init__(self):
        self.last_timeout = None

    async def post(self, *args, **kwargs):
        self.last_timeout = kwargs.get("timeout")
        return "ok"

    async def aclose(self):
        return None


def test_timeout_floor_raises_default_only():
    inner = _RecordingClient()
    client = _TimeoutFloorClient(inner, floor=86400)

    async def run():
        await client.post("url", timeout=_HARBOR_DEFAULT_HTTP_TIMEOUT)
        assert inner.last_timeout == 86400, "default timeout not raised to floor"
        await client.post("url", timeout=30)  # explicit short timeout preserved
        assert inner.last_timeout == 30

    asyncio.run(run())


def test_timeout_floor_proxies_other_attrs():
    inner = _RecordingClient()
    inner.marker = 123
    client = _TimeoutFloorClient(inner, floor=86400)
    assert client.marker == 123  # __getattr__ passthrough


# --------------------------------------------------------------------------- #
# Dockerfile dep-layer parser
# --------------------------------------------------------------------------- #

def _write_dockerfile(text: str):
    from pathlib import Path
    import tempfile
    d = Path(tempfile.mkdtemp())
    df = d / "Dockerfile"
    df.write_text(text)
    return df


def test_dep_layers_none_for_bare_from_workdir():
    df = _write_dockerfile("FROM gcc:13\nWORKDIR /workspace\n")
    assert _dockerfile_dep_layers(df) is None


def test_dep_layers_single_run():
    df = _write_dockerfile(
        "FROM gcc:13\n"
        "RUN apt-get update && apt-get install -y python3\n"
    )
    frag = _dockerfile_dep_layers(df)
    assert frag is not None
    assert "%post" in frag
    assert "apt-get update && apt-get install -y python3" in frag
    assert "%files" not in frag


def test_dep_layers_multiline_run_continuation():
    df = _write_dockerfile(
        "FROM gcc:13\n"
        "RUN apt-get update && \\\n"
        "    apt-get install -y python3 nlohmann-json3-dev\n"
    )
    frag = _dockerfile_dep_layers(df)
    assert frag is not None
    # continuation folded into one line
    assert "apt-get update && apt-get install -y python3 nlohmann-json3-dev" in frag


def test_dep_layers_copy_becomes_files_and_mkdir():
    df = _write_dockerfile(
        "FROM golang:1.24\n"
        "COPY setup.py /app/\n"
    )
    frag = _dockerfile_dep_layers(df)
    assert frag is not None
    assert "%files" in frag
    assert "setup.py /app/" in frag
    assert "mkdir -p /app/" in frag


def test_dep_layers_env_becomes_environment():
    df = _write_dockerfile(
        "FROM rust:1.90\n"
        "ENV PATH=/usr/bin:$PATH\n"
        "RUN python3 --version\n"
    )
    frag = _dockerfile_dep_layers(df)
    assert frag is not None
    assert "%environment" in frag
    assert "export PATH=/usr/bin:$PATH" in frag
    assert "python3 --version" in frag


def test_dep_layers_comments_and_other_instructions_ignored():
    df = _write_dockerfile(
        "# a comment\n"
        "FROM node:22\n"
        "WORKDIR /app\n"
        "RUN npm install -g typescript\n"
        "USER node\n"
    )
    frag = _dockerfile_dep_layers(df)
    assert frag is not None
    assert "npm install -g typescript" in frag
    assert "USER" not in frag
    assert "WORKDIR" not in frag

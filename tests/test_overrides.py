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
)


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

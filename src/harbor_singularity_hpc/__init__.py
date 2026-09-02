"""harbor-singularity-hpc: a writable-rootfs Singularity environment for harbor
on FUSE-restricted HPC clusters.

Select it via harbor's import-path route (no harbor source is patched):

    --environment-import-path harbor_singularity_hpc.environment:SingularityWritableEnvironment
"""

from harbor_singularity_hpc.environment import SingularityWritableEnvironment

__all__ = ["SingularityWritableEnvironment"]
__version__ = "0.1.0"

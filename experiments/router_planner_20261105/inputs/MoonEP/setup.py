"""
moonep build script. Compiles the CUDA extension `moonep._C` (the pybind11
module living in csrc/bindings.cu) into the `moonep` package so it can be
imported as `from moonep import _C` after either:

  pip install -e .                              # editable install (preferred)

or, for a quick in-place build without installing:

  python setup.py build_ext --inplace

The .so lands at `moonep/_C.<abi-tag>.so`.
"""

import ast
import os
import re
import subprocess

from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))


def _normalize_local_part(value):
    """Normalize an environment-derived value for a PEP 440 local version."""
    return re.sub(r"[^a-zA-Z0-9]+", ".", value).strip(".")


def get_package_version():
    """Return a traceable wheel version for the current build environment."""
    init_path = os.path.join(REPO_ROOT, "moonep", "__init__.py")
    with open(init_path, encoding="utf-8") as init_file:
        version_match = re.search(
            r"^__version__\s*=\s*(.*)$", init_file.read(), re.MULTILINE
        )
    if version_match is None:
        raise RuntimeError("moonep.__version__ is not defined")
    public_version = ast.literal_eval(version_match.group(1))

    dev_release = os.getenv("MOONEP_DEV_RELEASE", "").strip()
    if dev_release:
        if not dev_release.isdigit():
            raise ValueError("MOONEP_DEV_RELEASE must be a non-negative integer")
        public_version = f"{public_version}.dev{int(dev_release)}"

    local_parts = []
    for env_name in ("MOONEP_BUILD_REVISION", "CI_COMMIT_SHORT_SHA"):
        revision = _normalize_local_part(os.getenv(env_name, "").strip())
        if revision:
            local_parts.append(revision)
            break

    if not local_parts:
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
            ).strip()
            if revision:
                local_parts.append(_normalize_local_part(revision))
        except (OSError, subprocess.CalledProcessError):
            pass

    local_parts.extend(
        [
            "step",
            f"torch{_normalize_local_part(torch.__version__.split('+', 1)[0])}",
        ]
    )
    if torch.version.cuda:
        local_parts.append(f"cu{torch.version.cuda.replace('.', '')}")

    return f"{public_version}+{'.'.join(local_parts)}"


nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-std=c++20",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-forward-unknown-to-host-compiler",
    "-Xcompiler=-Wno-psabi",
    "-Xcompiler=-fno-strict-aliasing",
    "-DNDEBUG",
    "-lineinfo",
    "-diag-suppress=3189",
    "-ftemplate-backtrace-limit=0",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]

cxx_flags = [
    "-O3",
    "-fno-strict-aliasing",
    "-Wno-psabi",
]


ext_modules = [
    CUDAExtension(
        name="moonep._C",
        sources=[
            os.path.join("csrc", "bindings.cu"),
        ],
        include_dirs=[
            os.path.join(REPO_ROOT, "csrc"),
        ],
        libraries=["cuda"],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    ),
]


setup(
    name="moonep",
    version=get_package_version(),
    packages=find_packages(include=["moonep", "moonep.*"]),
    install_requires=[
        "nvidia-cutlass-dsl==4.4.2",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)

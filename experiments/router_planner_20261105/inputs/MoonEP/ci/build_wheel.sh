#!/usr/bin/env bash

set -euo pipefail

container_venv="${MOONEP_CONTAINER_VENV:-/venv-3.10/bin/activate}"
deps_prefix="${MOONEP_PY_PREFIX:-/tmp/moonep-python-deps}"
cutlass_dsl_package="${MOONEP_CUTLASS_DSL_PACKAGE:-nvidia-cutlass-dsl==4.4.2}"

echo "Building MoonEP wheel in CI image: ${MOONEP_CI_IMAGE:-unknown}"
echo "CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}"
if [ -n "${MOONEP_DEV_RELEASE:-}" ]; then
  echo "Development release: .dev${MOONEP_DEV_RELEASE}"
fi

if [ -f "${container_venv}" ]; then
  # shellcheck disable=SC1090
  . "${container_venv}"
fi

python3 --version
python3 - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA: {torch.version.cuda}")
PY

rm -rf build dist
find . -maxdepth 1 -name '*.egg-info' -exec rm -rf {} +
rm -rf "${deps_prefix}"
mkdir -p "${deps_prefix}"

pip_args=()
if [ -n "${MOONEP_PIP_INDEX_URL:-}" ]; then
  pip_args+=(-i "${MOONEP_PIP_INDEX_URL}")
fi
if [ -n "${MOONEP_PIP_EXTRA_INDEX_URL:-}" ]; then
  pip_args+=(--extra-index-url "${MOONEP_PIP_EXTRA_INDEX_URL}")
fi
for host in ${MOONEP_PIP_TRUSTED_HOSTS:-}; do
  pip_args+=(--trusted-host "${host}")
done

python3 -m pip install \
  --no-cache-dir \
  --ignore-installed \
  --prefix "${deps_prefix}" \
  "${pip_args[@]}" \
  build ninja packaging "${cutlass_dsl_package}"

site_packages=$(MOONEP_PY_PREFIX="${deps_prefix}" python3 - <<'PY'
import os
import sysconfig

prefix = os.environ["MOONEP_PY_PREFIX"]
print(sysconfig.get_path("purelib", vars={"base": prefix, "platbase": prefix}))
PY
)
scripts_path=$(MOONEP_PY_PREFIX="${deps_prefix}" python3 - <<'PY'
import os
import sysconfig

prefix = os.environ["MOONEP_PY_PREFIX"]
print(sysconfig.get_path("scripts", vars={"base": prefix, "platbase": prefix}))
PY
)
export PATH="${scripts_path}:${PATH}"
export PYTHONPATH="${site_packages}${PYTHONPATH:+:${PYTHONPATH}}"
export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"

python3 -m build --wheel --no-isolation --outdir dist

wheel_path=$(find dist -maxdepth 1 -type f -name 'moonep-*.whl' -print -quit)
if [ -z "${wheel_path}" ]; then
  echo "MoonEP wheel was not produced" >&2
  exit 1
fi

python3 - "${wheel_path}" <<'PY'
import email.parser
import os
import sys
import zipfile

from packaging.version import Version

wheel_path = sys.argv[1]
with zipfile.ZipFile(wheel_path) as wheel:
    names = wheel.namelist()
    if not any(name.startswith("moonep/") and name.endswith(".so") for name in names):
        raise SystemExit("wheel does not contain moonep CUDA extension")
    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    if len(metadata_names) != 1:
        raise SystemExit("wheel does not contain package metadata")
    metadata = email.parser.BytesParser().parsebytes(wheel.read(metadata_names[0]))

version = Version(metadata["Version"])
release_tag = os.environ.get("CI_COMMIT_TAG", "")
if release_tag:
    tagged_version = Version(release_tag.removeprefix("v"))
    if version.public != tagged_version.public:
        raise SystemExit(
            f"wheel public version {version.public} does not match tag {release_tag}"
        )
print(f"Validated wheel: {wheel_path}")
PY

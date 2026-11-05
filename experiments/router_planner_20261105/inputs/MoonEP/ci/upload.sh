#!/usr/bin/env bash

set -euo pipefail

: "${MOONEP_DEVPI_SERVER:?MOONEP_DEVPI_SERVER is required}"
: "${MOONEP_DEVPI_INDEX:?MOONEP_DEVPI_INDEX is required}"
: "${MOONEP_DEVPI_USER:?MOONEP_DEVPI_USER must be a protected CI variable}"
: "${MOONEP_DEVPI_PASSWORD:?MOONEP_DEVPI_PASSWORD must be a protected and masked CI variable}"

shopt -s nullglob
wheels=(dist/moonep-*.whl)
if [ "${#wheels[@]}" -eq 0 ]; then
  echo "No MoonEP wheels found under dist/" >&2
  exit 1
fi

python3 -m devpi use "${MOONEP_DEVPI_SERVER}"
python3 -m devpi login "${MOONEP_DEVPI_USER}" --password="${MOONEP_DEVPI_PASSWORD}"
python3 -m devpi use "${MOONEP_DEVPI_INDEX}"
python3 -m devpi upload "${wheels[@]}"

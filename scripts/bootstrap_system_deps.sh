#!/usr/bin/env bash
set -euo pipefail

TOOLS=(qpdf ghostscript ffmpeg findutils miller)
BINARIES=(qpdf gs ffmpeg find mlr)

mode="install"
if [[ "${1:-}" == "--check" ]]; then
  mode="check"
fi

missing_packages=()
missing_binaries=()
for i in "${!BINARIES[@]}"; do
  if ! command -v "${BINARIES[$i]}" >/dev/null 2>&1; then
    missing_binaries+=("${BINARIES[$i]}")
    missing_packages+=("${TOOLS[$i]}")
  fi
done

if [[ "${#missing_binaries[@]}" -eq 0 ]]; then
  echo "All required system tools are installed: ${BINARIES[*]}"
  exit 0
fi

echo "Missing system tools: ${missing_binaries[*]}"
echo "Install packages: ${missing_packages[*]}"

if [[ "$mode" == "check" ]]; then
  exit 1
fi

if [[ "$(id -u)" -eq 0 ]]; then
  apt-get update
  apt-get install -y "${missing_packages[@]}"
else
  sudo apt-get update
  sudo apt-get install -y "${missing_packages[@]}"
fi

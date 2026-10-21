#!/usr/bin/env bash
set -euo pipefail

# Preferred installer for the server. It creates the conda environment and
# caches under /home/s414f/s414f1 by default to avoid filling the system disk.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/setup_pada_env.sh"

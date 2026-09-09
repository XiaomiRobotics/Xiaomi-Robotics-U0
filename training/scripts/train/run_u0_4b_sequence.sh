#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CONFIG="${CONFIG:-${SCRIPT_DIR}/../../configs/train/u0_4b_sequence.yaml}"
exec bash "${SCRIPT_DIR}/run_u0_4b.sh" "$@"

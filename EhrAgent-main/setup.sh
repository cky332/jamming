#!/usr/bin/env bash
# Bootstrap script for jamming×MINJA × EHRAgent on Linux + Anaconda.
#
# Usage:
#   bash setup.sh                # create env, install deps, run offline tests
#   bash setup.sh smoke          # also run Phase 0 smoke (needs SILICONFLOW_API_KEY)
#
# Idempotent: re-running on an existing env updates deps but doesn't recreate.

set -euo pipefail

ENV_NAME="${ENV_NAME:-jamming-ehragent}"
PY_VER="3.10"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$THIS_DIR"

# --- conda detection ---
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found in PATH. Install Anaconda/Miniconda first." >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- env create / activate ---
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] env '$ENV_NAME' exists — activating + updating deps"
    conda activate "$ENV_NAME"
    conda env update -n "$ENV_NAME" -f environment.yml --prune
else
    echo "[setup] creating conda env '$ENV_NAME' (python=$PY_VER)"
    conda env create -n "$ENV_NAME" -f environment.yml
    conda activate "$ENV_NAME"
fi

echo
echo "[setup] python: $(which python)  $(python --version)"
echo

# --- sanity import check ---
echo "[setup] verifying core imports..."
python - <<'PY'
import sys
mods = [
    ("autogen",            "pyautogen"),
    ("openai",             "openai"),
    ("pandas",             "pandas"),
    ("numpy",              "numpy"),
    ("Levenshtein",        "python-Levenshtein"),
    ("sentence_transformers", "sentence-transformers"),
    ("jsonlines",          "jsonlines"),
    ("termcolor",          "termcolor"),
]
missing = []
for mod, pkg in mods:
    try:
        __import__(mod)
        print(f"  OK   {pkg}")
    except Exception as e:
        print(f"  MISS {pkg}: {e}")
        missing.append(pkg)
if missing:
    print(f"\n[setup] missing packages: {missing}", file=sys.stderr)
    sys.exit(1)
PY

# --- offline tests (no API) ---
echo
echo "[setup] running offline tests (no API needed)..."
python -m attack.tests.test_offline_components
echo
python -m attack.tests.test_phase0
echo

# --- MiniLM weights pre-download (if HF reachable) ---
echo "[setup] pre-downloading MiniLM weights (or detecting offline)..."
python - <<'PY'
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"  OK   MiniLM cached at {m._first_module().auto_model.config._name_or_path}")
except Exception as e:
    print(f"  WARN MiniLM download failed: {type(e).__name__}: {str(e)[:200]}")
    print("       Falls back to Levenshtein at runtime; main retriever degraded.")
PY

# --- env var check ---
echo
if [[ -n "${SILICONFLOW_API_KEY:-}" ]]; then
    echo "[setup] SILICONFLOW_API_KEY is set — Phase 1/2 should run."
else
    echo "[setup] SILICONFLOW_API_KEY is NOT set."
    echo "        cp .env.example .env  # then edit and: source .env (or export manually)"
fi

# --- optional: run smoke ---
if [[ "${1:-}" == "smoke" ]]; then
    if [[ -z "${SILICONFLOW_API_KEY:-}" ]]; then
        echo "[setup] cannot run smoke without SILICONFLOW_API_KEY." >&2
        exit 2
    fi
    echo
    echo "[setup] running Phase 0 smoke (live DeepSeek-V3.2 call)..."
    python -m attack.cli smoke
fi

echo
echo "=============================================================="
echo "setup complete."
echo "Next steps:"
echo "  1. conda activate $ENV_NAME"
echo "  2. cp .env.example .env  (then add your SILICONFLOW_API_KEY)"
echo "  3. source .env  (or: export SILICONFLOW_API_KEY=sk-...)"
echo "  4. python -m attack.cli smoke      # ~1 min, verify API + agent round-trip"
echo "  5. python -m attack.cli inject --victim 30789"
echo "  6. python -m attack.cli evaluate --victim 30789"
echo "=============================================================="

#!/usr/bin/env bash
set -euo pipefail

LIMIT="${1:-100}"

cd "$(dirname "$0")/.."

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

pip install -q -r requirements.txt

python - <<'PY'
from src.utils.db import bootstrap_sqlite, bootstrap_duckdb
bootstrap_sqlite(); bootstrap_duckdb()
print('DB OK')
PY

python ./main.py --demo --limit "$LIMIT"

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests -q

echo "✅✅✅ ALL TESTS PASSED — SYSTEM READY FOR BUILD 🚀"

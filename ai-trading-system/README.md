# AI Trading System Scaffold

This repository contains the scaffold for the AI Trading System described in `Docs/system_build_plan.md`. The implementation is staged according to the generated phase blueprint in `Docs/phase_blueprint.md`.

## Getting Started

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy the project-level `.env` file into this directory to provide runtime configuration.
4. Bootstrap the databases via the helper script: `python scripts/bootstrap_databases.py`.
5. Run `python main.py --demo --dry-run --today` to execute a dry-run once the system is implemented.

The bootstrap script provisions the SQLite and DuckDB schemas required for AI selections,
risk monitoring, technical indicators, and news ingestion, so downstream phases can
immediately persist data during development.

Refer to the documentation under `docs/` for detailed architecture notes as implementation progresses.

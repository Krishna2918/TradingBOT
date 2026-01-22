"""Helper script to bootstrap project databases."""
from src.utils import db


def main() -> None:
    db.bootstrap_all()
    print("Database bootstrap complete")
    print("  SQLite:", db.SQLITE_PATH)
    print("  DuckDB:", db.DUCKDB_PATH)


if __name__ == "__main__":
    main()

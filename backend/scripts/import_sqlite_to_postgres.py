"""Repeatable, non-destructive import of the existing SQLite workspace."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.dialects.postgresql import insert

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
from app import config as _config
from app.models import metadata

ORDER = ("users", "fields", "refresh_sessions", "observations", "satellite_acquisitions",
         "processing_jobs", "predictions", "job_outbox")


def normalize_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def migrate(source: Path, target_url: str, allow_nonempty: bool = False) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"SQLite source does not exist: {source}")
    target = create_engine(normalize_url(target_url), pool_pre_ping=True)
    # SQLite on Windows accepts forward-slash drive URIs (file:C:/...), while
    # Path.as_uri() yields file:///C:/ which some bundled SQLite builds reject.
    src = sqlite3.connect(f"file:{source.resolve().as_posix()}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    report = {"source": str(source), "tables": {}, "errors": []}
    try:
        inspector = inspect(target)
        missing = [name for name in ORDER if name not in inspector.get_table_names()]
        if missing:
            raise RuntimeError(f"Target schema is incomplete; run Alembic first (missing: {', '.join(missing)}).")
        with target.begin() as conn:
            counts = {name: conn.execute(text(f'SELECT COUNT(*) FROM "{name}"')).scalar_one()
                      for name in ORDER if name in inspector.get_table_names()}
            occupied = {name: count for name, count in counts.items() if count}
            if occupied and not allow_nonempty:
                raise RuntimeError("Target has existing rows; import stopped without changes. Use --allow-nonempty for conflict-safe merge.")
            for table_name in ORDER:
                source_tables = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if table_name not in source_tables:
                    continue
                table = metadata.tables[table_name]
                copied = skipped = 0
                for row in src.execute(f'SELECT * FROM "{table_name}"'):
                    values = {key: row[key] for key in row.keys() if key in table.c}
                    try:
                        with conn.begin_nested():
                            result = conn.execute(insert(table).values(**values).on_conflict_do_nothing())
                        if result.rowcount:
                            copied += 1
                        else:
                            skipped += 1
                    except Exception as exc:  # don't put row values (password/token hashes) in an import report
                        report["errors"].append({"table": table_name,
                            "row_id": str(values.get("id", "refresh_session" if table_name == "refresh_sessions" else "unknown")),
                            "error_type": type(exc).__name__})
                report["tables"][table_name] = {"inserted": copied, "already_present": skipped}
    finally:
        src.close()
        target.dispose()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(os.getenv("FASAL_DATABASE_PATH", "backend/data/fasal_sanket.sqlite3")))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="PostgreSQL URL; never pass via a shell history on shared machines")
    parser.add_argument("--allow-nonempty", action="store_true", help="Merge by stable primary/unique keys without overwriting target rows")
    parser.add_argument("--report", type=Path, default=Path("sqlite_import_report.json"))
    args = parser.parse_args()
    if not args.database_url:
        parser.error("Set DATABASE_URL in the environment or provide --database-url.")
    report = migrate(args.source, args.database_url, args.allow_nonempty)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"tables": report["tables"], "error_count": len(report["errors"]), "report": str(args.report)}, indent=2))
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

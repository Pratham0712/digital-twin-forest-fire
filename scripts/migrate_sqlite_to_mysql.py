"""
One-time copy of the local SQLite database (data/digital_twin.db) into the
database named by DATABASE_URL (e.g. a hosted MySQL instance).

    set DATABASE_URL=mysql+pymysql://USER:PASSWORD@HOST:PORT/DBNAME   (Windows)
    python scripts/migrate_sqlite_to_mysql.py

Creates any missing tables, then copies every row of every table, keeping the
original ids so alerts still point at their snapshots. Tables that already
contain rows in the target are skipped (safe to re-run).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, func, insert, inspect, select, text

from src.storage import database as db


def main():
    target_url = db.database_url()
    if target_url.startswith("sqlite"):
        sys.exit("DATABASE_URL is not set - point it at the target MySQL database first.")
    if not db.DB_PATH.exists():
        sys.exit(f"No local SQLite database at {db.DB_PATH} - nothing to migrate.")

    source = create_engine(f"sqlite:///{db.DB_PATH}")
    target = db.get_engine()
    db.metadata.create_all(target, checkfirst=True)
    source_tables = set(inspect(source).get_table_names())

    # parents before children (alerts_log references snapshots)
    for table in db.metadata.sorted_tables:
        if table.name not in source_tables:
            print(f"{table.name:<13} not in source, skipped")
            continue
        with target.connect() as tconn:
            existing = tconn.execute(select(func.count()).select_from(table)).scalar_one()
        if existing:
            print(f"{table.name:<13} target already has {existing} rows, skipped")
            continue
        src_cols = {c["name"] for c in inspect(source).get_columns(table.name)}
        cols = [c for c in table.columns if c.name in src_cols]
        with source.connect() as sconn:
            rows = [dict(r._mapping) for r in sconn.execute(select(*[table.c[c.name] for c in cols]))]
        if rows:
            with target.begin() as tconn:
                for i in range(0, len(rows), 1000):
                    tconn.execute(insert(table), rows[i:i + 1000])
                if target.dialect.name == "mysql":
                    next_id = max(r["id"] for r in rows) + 1
                    tconn.execute(text(f"ALTER TABLE {table.name} AUTO_INCREMENT = {next_id}"))
        print(f"{table.name:<13} copied {len(rows)} rows")

    print(f"\nDone. Dashboard will now use {db.backend_name()} whenever DATABASE_URL is set.")


if __name__ == "__main__":
    main()

"""
db_explorer.py - show the whole database live (for the project review).

    python scripts/db_explorer.py schema            # tables, columns, types, keys, indexes
    python scripts/db_explorer.py counts            # rows per table
    python scripts/db_explorer.py sample snapshots  # first rows of a table (password data hidden)
    python scripts/db_explorer.py sql "SELECT severity, COUNT(*) FROM alerts_log GROUP BY severity"
    python scripts/db_explorer.py crud              # full Create / Read / Update / Delete demo

Works on whatever DATABASE_URL points to (.env), MySQL or the SQLite fallback.
The crud demo only touches one row that it creates itself and removes again.
"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import pandas as pd  # noqa: E402
from sqlalchemy import inspect, text  # noqa: E402

from src.storage.database import backend_name, get_connection, get_engine, init_db  # noqa: E402

HIDDEN = {"password_hash", "salt"}


def show(df: pd.DataFrame):
    print(df.to_string(index=False) if len(df) else "(no rows)")


def cmd_schema():
    insp = inspect(get_engine())
    print(f"Backend: {backend_name()}\n")
    for t in sorted(insp.get_table_names()):
        print(f"TABLE {t}")
        pk = set(insp.get_pk_constraint(t).get("constrained_columns", []))
        rows = [{"column": c["name"], "type": str(c["type"]),
                 "null": "YES" if c["nullable"] else "NO",
                 "key": "PK" if c["name"] in pk else ""} for c in insp.get_columns(t)]
        show(pd.DataFrame(rows))
        for fk in insp.get_foreign_keys(t):
            print(f"  FK {fk['constrained_columns']} -> {fk['referred_table']}{fk['referred_columns']}")
        for ix in insp.get_indexes(t):
            print(f"  INDEX {ix['name']} on {ix['column_names']}{' UNIQUE' if ix.get('unique') else ''}")
        print()


def cmd_counts():
    insp = inspect(get_engine())
    with get_connection() as c:
        rows = [{"table": t, "rows": c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()}
                for t in sorted(insp.get_table_names())]
    show(pd.DataFrame(rows))


def cmd_sample(table: str, n: int):
    if table not in inspect(get_engine()).get_table_names():
        sys.exit(f"Unknown table '{table}'. Run 'schema' to list tables.")
    with get_connection() as c:
        df = pd.DataFrame(c.execute(text(f"SELECT * FROM {table} ORDER BY id DESC LIMIT {int(n)}")).mappings().all())
    df = df.drop(columns=[c for c in df.columns if c in HIDDEN], errors="ignore")
    show(df)


def cmd_sql(query: str):
    if not query.lstrip().lower().startswith(("select", "show", "describe", "explain")):
        sys.exit("Only read queries (SELECT/SHOW/DESCRIBE/EXPLAIN) are allowed here.")
    with get_connection() as c:
        res = c.execute(text(query))
        show(pd.DataFrame(res.mappings().all()))


def cmd_crud():
    """Create -> Read -> Update -> Delete on one activity_log row."""
    from src.storage.database import activity_log
    from datetime import datetime, timezone
    from sqlalchemy import insert, select, update, delete
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as c:
        new_id = c.execute(insert(activity_log).values(
            timestamp_utc=now, event_type="review_demo", actor="reviewer",
            region_name="Demo", detail="row created for the CRUD demo")).inserted_primary_key[0]
    print(f"CREATE  inserted activity_log row id={new_id}")
    with get_connection() as c:
        row = c.execute(select(activity_log).where(activity_log.c.id == new_id)).mappings().one()
    print("READ   ", dict(row))
    with get_connection() as c:
        c.execute(update(activity_log).where(activity_log.c.id == new_id)
                  .values(detail="row UPDATED during the CRUD demo"))
        row = c.execute(select(activity_log.c.detail).where(activity_log.c.id == new_id)).scalar()
    print(f"UPDATE  detail is now: {row!r}")
    with get_connection() as c:
        c.execute(delete(activity_log).where(activity_log.c.id == new_id))
        left = c.execute(select(activity_log).where(activity_log.c.id == new_id)).first()
    print(f"DELETE  row removed: {'yes' if left is None else 'NO - still there'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("schema")
    sub.add_parser("counts")
    s = sub.add_parser("sample"); s.add_argument("table"); s.add_argument("-n", type=int, default=10)
    q = sub.add_parser("sql"); q.add_argument("query")
    sub.add_parser("crud")
    a = ap.parse_args()
    init_db()
    if a.cmd == "schema":
        cmd_schema()
    elif a.cmd == "counts":
        cmd_counts()
    elif a.cmd == "sample":
        cmd_sample(a.table, a.n)
    elif a.cmd == "sql":
        cmd_sql(a.query)
    else:
        cmd_crud()


if __name__ == "__main__":
    main()

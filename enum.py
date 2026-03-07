#!/usr/bin/env python3
 

from pathlib import Path
from typing import List, Tuple, Set, Iterable
import json
import psycopg2
from psycopg2.extras import DictCursor

CONFIG = {
    "DB": {
        "DSN": None,
        "HOST": "127.0.0.1",
        "PORT": 5432,
        "DBNAME": "postgres",
        "USER": "postgres",
        "PASSWORD": "",
    },
    "SCHEMA": "public",
    "PAIRABLE_JSON_PATH": "pairable.json",
    "QUERIES_SQL_PATH": "queries.sql",
    "INLINE_QUERIES": [],
    "OUTPUT_PATH": "enum_plan.txt",
    "USE_EXPLAIN_ANALYZE": False,
    "STATEMENT_TIMEOUT_MS": 0,
}

BANNER = "#" * 80


def load_pairs(path: Path) -> List[Tuple[str, str]]:
    
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        if "pairable" in obj:
            pairs = obj["pairable"]
        elif "pairs" in obj:
            pairs = obj["pairs"]
        else:
            raise ValueError("JSON dict must include 'pairable' or 'pairs' field.")
    elif isinstance(obj, list):
        pairs = obj
    else:
        raise ValueError("Unsupported pairable.json structure.")

    norm = set()
    out: List[Tuple[str, str]] = []
    for p in pairs:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"Invalid pair: {p}")
        a, b = str(p[0]).strip(), str(p[1]).strip()
        key = tuple(sorted((a, b)))
        if key not in norm:
            norm.add(key)
            out.append((a, b))
    return out


def all_tables_from_pairs(pairs: Iterable[Tuple[str, str]]) -> Set[str]:
    s: Set[str] = set()
    for a, b in pairs:
        s.add(a)
        s.add(b)
    return s


def read_queries_from_file(path: Path) -> List[str]:
    sql = Path(path).read_text(encoding="utf-8")
    parts = [p.strip() for p in sql.split(";")]
    queries = [p for p in parts if p]
    if not queries:
        raise ValueError(f"No SQL statements found in {path} (separated by semicolons).")
    return [q + ";" for q in queries]


def normalize_inline_queries(qs: List[str]) -> List[str]:
    out: List[str] = []
    for q in qs:
        s = q.strip()
        if not s:
            continue
        if not s.endswith(";"):
            s += ";"
        out.append(s)
    return out


def exec_silent(cur, stmt: str, params: tuple = ()):
    try:
        cur.execute(stmt, params)
    except Exception:
        cur.connection.rollback()
        cur.connection.autocommit = True


def ensure_search_path(cur, schema: str = None):
    if schema:
        cur.execute("SET search_path TO %s, public;", (schema,))


def qualify_name(tbl: str, schema: str) -> str:
    if schema and "." not in tbl:
        return f"{schema}.{tbl}"
    return tbl


def drop_from_columnar(cur, tables: Iterable[str], schema: str = None):
    for t in tables:
        rel = qualify_name(t, schema)
        exec_silent(cur, "SELECT google_columnar_engine_drop(relation => %s);", (rel,))


def add_to_columnar(cur, tables: Iterable[str], schema: str = None):
    for t in tables:
        rel = qualify_name(t, schema)
        exec_silent(cur, "SELECT google_columnar_engine_add(relation => %s);", (rel,))


def explain_query(cur, query: str, analyze: bool = False) -> str:
    suffix = "ANALYZE, " if analyze else ""
    cur.execute(f"EXPLAIN ({suffix}COSTS, VERBOSE, FORMAT TEXT) " + query)
    rows = cur.fetchall()
    return "\n".join(r[0] for r in rows)


def chunk_header(pair: Tuple[str, str], idx: int, total: int) -> str:
    a, b = pair
    return f"{BANNER}\n# Pair {idx}/{total}: ({a}, {b})\n{BANNER}\n"


def main():
    cfg = CONFIG
    pairs = load_pairs(Path(cfg["PAIRABLE_JSON_PATH"]))
    all_tabs = all_tables_from_pairs(pairs)
    if cfg["INLINE_QUERIES"]:
        queries = normalize_inline_queries(cfg["INLINE_QUERIES"])
    else:
        queries = read_queries_from_file(Path(cfg["QUERIES_SQL_PATH"]))
    db = cfg["DB"]
    if db.get("DSN"):
        conn = psycopg2.connect(db["DSN"])
    else:
        conn = psycopg2.connect(
            host=db["HOST"],
            port=db["PORT"],
            dbname=db["DBNAME"],
            user=db["USER"],
            password=db["PASSWORD"],
        )
    conn.autocommit = True

    out_path = Path(cfg["OUTPUT_PATH"]).resolve()

    with conn, conn.cursor(cursor_factory=DictCursor) as cur, out_path.open("w", encoding="utf-8") as fout:
        ensure_search_path(cur, cfg["SCHEMA"])
        cur.execute("SET client_min_messages = WARNING;")
        timeout = int(cfg.get("STATEMENT_TIMEOUT_MS", 0) or 0)
        if timeout >= 0:
            cur.execute("SET statement_timeout = %s;", (timeout,))

        total = len(pairs)
        drop_from_columnar(cur, all_tabs, cfg["SCHEMA"])

        for i, pair in enumerate(pairs, 1):
            a, b = pair
            drop_from_columnar(cur, all_tabs - set(pair), cfg["SCHEMA"])
            add_to_columnar(cur, pair, cfg["SCHEMA"])
            fout.write(chunk_header(pair, i, total))
            for qi, q in enumerate(queries, 1):
                fout.write(f"-- Query {qi} --------------------------------------------------\n")
                fout.write(q.strip() + "\n\n")
                try:
                    plan = explain_query(cur, q, analyze=cfg["USE_EXPLAIN_ANALYZE"])
                except Exception as e:
                    conn.rollback()
                    conn.autocommit = True
                    plan = f"[ERROR] {type(e).__name__}: {e}"
                fout.write(plan + "\n\n")
        drop_from_columnar(cur, all_tabs, cfg["SCHEMA"])

    print(f"Done. Plans written to: {out_path}")


if __name__ == "__main__":
    main()
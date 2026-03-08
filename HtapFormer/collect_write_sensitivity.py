
"""
Online / offline workflow for computing HTAP write-sensitivity coefficients (α_i, α_u, α_d).

Online mode:
    1. Load SQL templates from the specified directory (select.sql / insert.sql / update.sql / delete.sql) or custom SQL files.
    2. Connect to PostgreSQL via psycopg2 and run EXPLAIN (ANALYZE, FORMAT JSON).
    3. Parse execution time (Execution Time in milliseconds) and append rows to a CSV.
    4. Reuse profile_write_sensitivity.py utilities to derive α_i, α_u, α_d and write them as JSON.

Offline mode:
    - If write_sensitivity.csv already exists (collected via psql, a previous run, or other profiling),
      pass --from-csv-only to skip database work and compute α directly from that CSV.

Examples:
    # online mode: setup + collection + calculation
    python collect_write_sensitivity.py --host 127.0.0.1 --port 5432 --dbname YOURDB --user YOURUSERNAME --password YOURPASSWORD

    # offline mode: reuse an existing CSV
    python collect_write_sensitivity.py --from-csv-only

Notes:
    - Ensure PostgreSQL is accessible and the target database exists; this script will create tables and seed data automatically.
    - EXPLAIN ANALYZE runs the statements for real, so INSERT/UPDATE/DELETE will change data unless you manage rollbacks.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd
import psycopg2

from profile_write_sensitivity import (
    calculate_operation_cost,
    compute_alpha_values,
    save_results,
)


DEFAULT_SQL_FILES = {
    "SELECT": "select.sql",
    "INSERT": "insert.sql",
    "UPDATE": "update.sql",
    "DELETE": "delete.sql",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SQL profiling and compute HTAP write-sensitivity coefficients"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--dbname", type=str, default=None, help="Database name (required for online mode)")
    parser.add_argument("--user", type=str, default=None, help="Database user (required for online mode)")
    parser.add_argument("--password", type=str, default=None, help="Database password (required for online mode)")

    parser.add_argument(
        "--from-csv-only",
        action="store_true",
        help="Use an existing CSV to compute α ",
    )
    parser.add_argument(
        "--csv-input",
        type=str,
        default=os.path.join("data", "profiling", "write_sensitivity.csv"),
        help="CSV file to read in offline mode (default: data/profiling/write_sensitivity.csv)",
    )
    parser.add_argument(
        "--schema-sql",
        type=str,
        default=os.path.join("data", "profiling", "sql", "schema.sql"),
        help="SQL script for schema creation and data seeding (default: data/profiling/sql/schema.sql)",
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip schema/data initialization (schema.sql runs by default)",
    )
    parser.add_argument(
        "--sql-dir",
        type=str,
        default=os.path.join("data", "profiling", "sql"),
        help="Directory that stores select.sql / insert.sql / update.sql / delete.sql templates",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="How many times to repeat each SQL statement (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs per SQL (not recorded, default: 1)",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=os.path.join("data", "profiling", "write_sensitivity.csv"),
        help="Destination CSV for online profiling results",
    )
    parser.add_argument(
        "--alpha-output",
        type=str,
        default=os.path.join("checkpoints", "write_sensitivity.json"),
        help="Output JSON for α_i / α_u / α_d",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="latency_ms",
        help="Metric column used for alpha calculation (default: latency_ms)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum samples per operation before issuing a warning (default: 5)",
    )
    parser.add_argument(
        "--no-rollback",
        action="store_true",
        help="Disable automatic rollback after each iteration to persist writes",
    )
    parser.add_argument(
        "--sql-files",
        type=str,
        nargs="*",
        default=None,
        help="Custom SQL mapping, e.g. INSERT=my_insert.sql",
    )
    return parser.parse_args()


def execute_sql_script(conn, script_path: str):
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Schema script not found: {script_path}")

    with open(script_path, "r", encoding="utf-8") as f:
        sql_text = f.read()

    statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]
    if not statements:
        print(f"[INFO] Schema script {script_path} is empty, skip execution.")
        return

    print(f"[INFO] Running schema script ({len(statements)} statements): {script_path}")
    with conn.cursor() as cursor:
        for stmt in statements:
            cursor.execute(stmt)
    conn.commit()
    print("[OK] Schema initialization finished.")


def load_sql_templates(sql_dir: str, custom_files=None):
    files = DEFAULT_SQL_FILES.copy()
    if custom_files:
        for item in custom_files:
            if "=" not in item:
                raise ValueError(f"Invalid sql-files argument: {item}, expected op=filename")
            op, filename = item.split("=", 1)
            op = op.strip().upper()
            if op not in files:
                raise ValueError(f"Unknown operation {op}, available keys: {list(files.keys())}")
            files[op] = filename.strip()

    sql_map = {}
    for op, filename in files.items():
        path = os.path.join(sql_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"SQL file for {op} not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            sql = f.read().strip()
            if not sql:
                raise ValueError(f"{path} is empty, please provide SQL text for {op}.")
            sql_map[op] = sql
    return sql_map


def run_explain_analyze(cursor, sql: str):
    """
    Execute EXPLAIN (ANALYZE, FORMAT JSON) and return the execution time (ms).
    """
    explain_sql = f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}"
    cursor.execute(explain_sql)
    result = cursor.fetchone()
    if result is None or len(result) == 0:
        raise RuntimeError("EXPLAIN ANALYZE returned no result.")
    # psycopg2 maps JSON to Python objects automatically
    plan_list = result[0]
    if isinstance(plan_list, str):
        plan_list = json.loads(plan_list)
    root = plan_list[0]
    plan = root.get("Plan", {})
    execution_time = root.get("Execution Time") or plan.get("Actual Total Time")
    if execution_time is None:
        raise RuntimeError(f"Execution Time not found in EXPLAIN output: {json.dumps(root, indent=2)[:500]}")
    return execution_time, plan


def collect_metrics(conn, sql_map, iterations, warmup, rollback_each=True):
    rows = []
    with conn.cursor() as cursor:
        for op, sql in sql_map.items():
            print(f"[INFO] Profiling {op} ...")

            # Warm-up runs
            for _ in range(warmup):
                cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
                if rollback_each:
                    conn.rollback()
                else:
                    conn.commit()

            for it in range(1, iterations + 1):
                exec_time, plan = run_explain_analyze(cursor, sql)
                rows.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "op_type": op,
                        "latency_ms": exec_time,
                        "iteration": it,
                        "sql_snippet": sql[:120].replace("\n", " "),
                    }
                )
                print(f"    Iter {it}: {exec_time:.2f} ms")
                if rollback_each:
                    conn.rollback()
                else:
                    conn.commit()
    return rows


def save_csv(rows, csv_output):
    os.makedirs(os.path.dirname(csv_output), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_output, index=False)
    print(f"[OK] Profiling data written to {csv_output}")
    return df


def main():
    args = parse_args()

    if args.from_csv_only:
        if not os.path.exists(args.csv_input):
            raise FileNotFoundError(f"[from-csv-only] CSV not found: {args.csv_input}")
        df = pd.read_csv(args.csv_input)
        print(f"[INFO] Loaded {len(df)} rows from CSV: {args.csv_input}")
    else:
        missing_db_args = [name for name in ("dbname", "user", "password") if getattr(args, name) is None]
        if missing_db_args:
            missing_str = ", ".join(missing_db_args)
            raise ValueError(f"Online mode requires DB arguments: {missing_str}")

        sql_map = load_sql_templates(args.sql_dir, args.sql_files)

        conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=args.password,
        )

        try:
            if not args.skip_schema:
                execute_sql_script(conn, args.schema_sql)

            rows = collect_metrics(
                conn,
                sql_map,
                iterations=args.iterations,
                warmup=args.warmup,
                rollback_each=not args.no_rollback,
            )
        finally:
            conn.close()

        df = save_csv(rows, args.csv_output)

    costs, warnings = calculate_operation_cost(df, metric=args.metric, min_samples=args.min_samples)
    for w in warnings:
        print(w)
    alpha_values = compute_alpha_values(costs)
    save_results(args.alpha_output, alpha_values, metric=args.metric)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


#!/usr/bin/env python3
 

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pulp


 

def load_sizes(path: Path) -> Tuple[Dict[str, float], float]:
    
    obj = json.loads(Path(path).read_text())
    if "columns" not in obj:
        raise ValueError("size.json 需包含字段 'columns'。")
    cols = obj["columns"]
    sizes = {}
    for c, v in cols.items():
        if isinstance(v, dict):
            if "size" not in v:
                raise ValueError(f"size.json 中列 {c} 缺少 'size' 字段。")
            sizes[c] = float(v["size"])
        else:
            sizes[c] = float(v)
    M = obj.get("memory_budget", None)
    return sizes, M


def load_benefit(path: Path):
    
    obj = json.loads(Path(path).read_text())
    data = obj.get("data", obj)

    benefit = {}
    requires = {}
    Q: Set[str] = set()
    T: Set[str] = set()

    for q, tables in data.items():
        if not isinstance(tables, dict):
            raise ValueError(f"benefit.json: 查询 {q} 的值应为对象（每个表一个条目）")
        Q.add(q)
        for t, entry in tables.items():
            T.add(t)
            if isinstance(entry, dict):
                if "benefit" not in entry or "columns" not in entry:
                    raise ValueError(
                        f"benefit.json: (q={q}, t={t}) 需包含 'benefit' 与 'columns'。"
                    )
                b = float(entry["benefit"])
                cols = list(entry["columns"])
            else:
                raise ValueError(
                    f"benefit.json: (q={q}, t={t}) 必须为对象，包含 benefit 与 columns。"
                )
            if len(cols) == 0:
                # 若查询 q 不涉及表 t，可不在 JSON 中给出该表；若给了条目，columns 不能空
                raise ValueError(f"benefit.json: (q={q}, t={t}) 的 columns 为空。")
            benefit[(q, t)] = b
            requires[(q, t)] = cols

    return benefit, requires, Q, T


def validate_columns(requires, sizes):
    
    missing = []
    for (q, t), cols in requires.items():
        for c in cols:
            if c not in sizes:
                missing.append((q, t, c))
    if missing:
        msgs = [f"(q={q}, t={t}, col={c})" for (q, t, c) in missing]
        raise ValueError(
            "以下 (查询, 表, 列) 在 benefit.json 中声明但未在 size.json 中给出大小：\n"
            + "\n".join(msgs)
        )


 

def build_and_solve_ilp(
    sizes: Dict[str, float],
    benefit: Dict[Tuple[str, str], float],
    requires: Dict[Tuple[str, str], List[str]],
    memory_budget: float,
    solver_name: str = "CBC",
    msg: bool = False,
):
    
    prob = pulp.LpProblem("ColumnStoreSelection", pulp.LpMaximize)

    columns = sorted(sizes.keys())
    x = pulp.LpVariable.dicts("x", columns, lowBound=0, upBound=1, cat="Binary")

    pairs = sorted(requires.keys())
    y = {
        (q, t): pulp.LpVariable(f"y_{q}_{t}", lowBound=0, upBound=1, cat="Binary")
        for (q, t) in pairs
    }

    prob += pulp.lpSum(benefit[(q, t)] * y[(q, t)] for (q, t) in pairs)

    prob += pulp.lpSum(sizes[c] * x[c] for c in columns) <= float(memory_budget), "Budget"

    for (q, t) in pairs:
        S = list(requires[(q, t)])
        for c in S:
            prob += y[(q, t)] <= x[c], f"CoverUpper_{q}_{t}_{c}"
        prob += (
            y[(q, t)] >= pulp.lpSum(x[c] for c in S) - (len(S) - 1)
        ), f"CoverLower_{q}_{t}"

    if solver_name.upper() == "CBC":
        solver = pulp.PULP_CBC_CMD(msg=msg)
    elif solver_name.upper() == "GLPK":
        solver = pulp.GLPK_CMD(msg=msg)
    else:
        solver = None

    status = prob.solve(solver)

    return prob, x, y, status


 

def main():
    ap.add_argument("--sizes", required=True, type=Path, help="path to size.json")
    ap.add_argument("--benefit", required=True, type=Path, help="path to benefit.json")
    ap.add_argument("-M", "--memory", type=float, default=None,
                    help="override memory budget in size.json")
    ap.add_argument("--solver", choices=["CBC", "GLPK"], default="CBC",
                    help="MILP solver to use (default CBC)")
    ap.add_argument("--out", type=Path, default=Path("plan.json"),
                    help="output plan json (selected columns & realized benefits)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    sizes, M_file = load_sizes(args.sizes)
    benefit, requires, Q, T = load_benefit(args.benefit)
    validate_columns(requires, sizes)

    M = args.memory if args.memory is not None else M_file
    if M is None:
        raise ValueError("未提供内存预算：请在 size.json 里加入 memory_budget，或用 -M 指定。")

    if args.verbose:
        print(f"[INFO] Columns: {len(sizes)}; QxT pairs: {len(requires)}; Budget M={M}")

    prob, x, y, status = build_and_solve_ilp(
        sizes=sizes,
        benefit=benefit,
        requires=requires,
        memory_budget=M,
        solver_name=args.solver,
        msg=args.verbose,
    )

    selected_cols = [c for c in x if x[c].value() >= 0.5]
    selected_size = sum(sizes[c] for c in selected_cols)

    plan = {
        "selected_columns": selected_cols,
        "selected_size": selected_size,
    }

    args.out.write_text(json.dumps(plan, ensure_ascii=False, indent=2))

    print("Selected columns:", ", ".join(selected_cols) if selected_cols else "(none)")
    print(f"Detailed plan saved to: {args.out}")


if __name__ == "__main__":
    main()
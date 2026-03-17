"""
Utilities for computing HTAP write-sensitivity coefficients (α_i, α_u, α_d).

This file is used by `collect_write_sensitivity.py`.

Expected input CSV/DF columns (minimum):
  - op_type: one of SELECT / INSERT / UPDATE / DELETE (case-insensitive accepted)
  - metric column (default: latency_ms): numeric latency/cost

Output JSON format (consumed by `HtapFormer.py`):
  {
    "alpha_i": ...,
    "alpha_u": ...,
    "alpha_d": ...,
    "metric": "latency_ms",
    "timestamp": "YYYY-mm-dd HH:MM:SS"
  }
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd


def _normalize_op(op: str) -> str:
    op = (op or "").strip().upper()
    # tolerate common variants
    if op in {"SEL", "READ"}:
        return "SELECT"
    if op in {"INS"}:
        return "INSERT"
    if op in {"UPD"}:
        return "UPDATE"
    if op in {"DEL"}:
        return "DELETE"
    return op


def calculate_operation_cost(
    df: pd.DataFrame, metric: str = "latency_ms", min_samples: int = 5
) -> Tuple[Dict[str, float], List[str]]:
  
    if df is None or len(df) == 0:
        raise ValueError("Input dataframe is empty; cannot compute write sensitivity.")

    if "op_type" not in df.columns:
        raise ValueError("Missing required column: op_type")
    if metric not in df.columns:
        raise ValueError(f"Missing required metric column: {metric}")

    tmp = df.copy()
    tmp["op_type"] = tmp["op_type"].astype(str).map(_normalize_op)
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.dropna(subset=[metric])

    if len(tmp) == 0:
        raise ValueError(f"All metric values are NaN after parsing: {metric}")

    costs: Dict[str, float] = {}
    warnings: List[str] = []

    for op, group in tmp.groupby("op_type"):
        if len(group) < min_samples:
            warnings.append(
                f"[WARN] op_type={op} has only {len(group)} samples (<{min_samples}); alpha may be unstable."
            )
        costs[op] = float(group[metric].mean())

    # Ensure keys exist if possible (best-effort)
    for op in ("SELECT", "INSERT", "UPDATE", "DELETE"):
        if op not in costs:
            warnings.append(f"[WARN] op_type={op} is missing in data; will fallback for alpha computation.")

    return costs, warnings


def compute_alpha_values(costs: Dict[str, float]) -> Dict[str, float]:
    """
    Compute α values following the paper's normalization:

        α_t = (C_t / C_ref) / Σ_{k∈{i,u,d}} (C_k / C_ref)

    where C_ref is the baseline read-only cost (e.g., SELECT), and t is one of
    INSERT/UPDATE/DELETE.

    Notes:
      - This makes α_i + α_u + α_d = 1 (when all three write costs are available/valid).
      - If some write ops are missing, we normalize over the available ones.
      - If none of INSERT/UPDATE/DELETE are available, fall back to uniform (1/3 each).
    """

    def _valid_positive(x) -> bool:
        try:
            return x is not None and float(x) > 0.0
        except Exception:
            return False

    select_cost = costs.get("SELECT")
    if not _valid_positive(select_cost):
        positives = [float(v) for v in costs.values() if _valid_positive(v)]
        select_cost = min(positives) if positives else 1.0

    select_cost = float(select_cost)
    eps = 1e-12

    # compute relative ratios to C_ref (SELECT)
    ratios: Dict[str, float] = {}
    for op in ("INSERT", "UPDATE", "DELETE"):
        v = costs.get(op)
        if _valid_positive(v):
            ratios[op] = float(v) / max(select_cost, eps)

    # normalize within write ops
    denom = sum(ratios.values())
    if denom <= eps:
        # No valid write costs; fall back to uniform weights
        return {"alpha_i": 1.0 / 3.0, "alpha_u": 1.0 / 3.0, "alpha_d": 1.0 / 3.0}

    def norm(op: str) -> float:
        return float(ratios.get(op, 0.0)) / denom

    return {"alpha_i": norm("INSERT"), "alpha_u": norm("UPDATE"), "alpha_d": norm("DELETE")}


def save_results(output_path: str, alpha_values: Dict[str, float], metric: str = "latency_ms", costs: Dict[str, float] | None = None):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    payload = {
        "alpha_i": float(alpha_values.get("alpha_i", 1.0)),
        "alpha_u": float(alpha_values.get("alpha_u", 1.0)),
        "alpha_d": float(alpha_values.get("alpha_d", 1.0)),
        "metric": metric,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if costs is not None:
        payload["costs"] = {k: float(v) for k, v in sorted(costs.items(), key=lambda kv: kv[0])}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[OK] α values written to {output_path}: "
          f"alpha_i={payload['alpha_i']:.4f}, alpha_u={payload['alpha_u']:.4f}, alpha_d={payload['alpha_d']:.4f}")


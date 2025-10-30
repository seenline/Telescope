#!/usr/bin/env python3

from __future__ import annotations
import json, itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ★ 直接 import 工具模块
from simplify_plan_json import simplify_plan_json
from plan_traversal import get_traversals_from_plan_json

CONFIG = {
    "ROW_PLAN_JSON_PATH": "row_plan.json",
    "PAIR_PRIORITY_JSON_PATH": "pair_priority.json",
    "COLUMNAR_SET": [
        # 例如： "orders", "lineitem", "customer"
    ],
    "OUT_JSON": "generated_join_order.json",
    "OUT_TXT": "generated_join_order.txt",
}

def _children(n: Dict[str, Any]) -> List[Dict[str, Any]]:
    ch = n.get("Plans") or []
    return [c for c in ch if isinstance(c, dict)] if isinstance(ch, list) else []

def _node_type(n: Dict[str, Any]) -> str:
    return str(n.get("Node Type", "")).lower()

def _is_join(n: Dict[str, Any]) -> bool:
    return "join" in _node_type(n)

def _leaf_label(n: Dict[str, Any]) -> Optional[str]:
    if "scan" not in _node_type(n):
        return None
    rel = n.get("Relation Name")
    if isinstance(rel, str) and rel:
        return rel
    alias = n.get("Alias")
    if isinstance(alias, str) and alias:
        return alias
    return None

def _root_from_simplified(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj.get("Plan", obj)
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj[0].get("Plan", obj[0])
    raise TypeError("Unsupported simplified JSON structure")

def _normalize_tbl(s: str) -> str:
    s = s.strip()
    if "." in s:
        s = s.split(".")[-1]
    return s.lower()

def _collect_leaves(root: Dict[str, Any]) -> List[Dict[str, Any]]:
    out, st = [], [root]
    while st:
        n = st.pop()
        ch = _children(n)
        if not ch:
            out.append(n)
        else:
            st.extend(reversed(ch))
    return out

def _one_leaf(node: Dict[str, Any]) -> Optional[str]:
    st = [node]
    while st:
        n = st.pop()
        ch = _children(n)
        if not ch:
            lab = _leaf_label(n)
            if lab:
                return _normalize_tbl(lab)
        else:
            st.extend(reversed(ch))
    return None

def _row_spanning_tree_edges(root: Dict[str, Any]) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    st = [root]
    while st:
        n = st.pop()
        ch = _children(n)
        if ch:
            st.extend(reversed(ch))
        if _is_join(n) and len(ch) >= 2:
            a = _one_leaf(ch[0]); b = _one_leaf(ch[1])
            if a and b and a != b:
                e = tuple(sorted((a, b)))
                if e not in edges:
                    edges.append(e)
    return edges

def _row_leaf_depth(root: Dict[str, Any]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    st = [(root, 0)]
    while st:
        n, depth = st.pop()
        ch = _children(n)
        if not ch:
            lab = _leaf_label(n)
            if lab:
                d[_normalize_tbl(lab)] = depth
        else:
            for c in ch:
                st.append((c, depth + 1))
    return d

class DSU:
    def __init__(self, nodes: Set[str]):
        self.parent = {x: x for x in nodes}
        self.rank = {x: 0 for x in nodes}
    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

def main():
    cfg = CONFIG

    row_raw = json.loads(Path(cfg["ROW_PLAN_JSON_PATH"]).read_text(encoding="utf-8"))
    row_simplified_str = simplify_plan_json(json.dumps(row_raw))
    root = _root_from_simplified(json.loads(row_simplified_str))

    leaves = _collect_leaves(root)
    tables_in_query: Set[str] = set()
    for lf in leaves:
        lab = _leaf_label(lf)
        if lab:
            tables_in_query.add(_normalize_tbl(lab))
    if not tables_in_query:
        raise RuntimeError("No scan leaves found in row plan.")

    row_edges = _row_spanning_tree_edges(root)
    row_neighbors: Dict[str, Set[str]] = {t: set() for t in tables_in_query}
    for u, v in row_edges:
        row_neighbors[u].add(v); row_neighbors[v].add(u)
    row_depth = _row_leaf_depth(root)

    pp = json.loads(Path(cfg["PAIR_PRIORITY_JSON_PATH"]).read_text(encoding="utf-8"))
    if isinstance(pp, dict) and "items" in pp:
        pp = pp["items"]
    pair2score: Dict[Tuple[str, str], float] = {}
    for it in pp:
        pr = it.get("pair") or it.get("tables")
        if not pr or len(pr) != 2:
            continue
        a, b = _normalize_tbl(pr[0]), _normalize_tbl(pr[1])
        key = tuple(sorted((a, b)))
        sc = float(it.get("score", 0.0))
        pair2score[key] = max(sc, pair2score.get(key, 0.0))

    CS = {_normalize_tbl(t) for t in cfg["COLUMNAR_SET"] if _normalize_tbl(t) in tables_in_query}

    cand = []
    for a, b in itertools.combinations(sorted(CS), 2):
        key = tuple(sorted((a, b)))
        if key in pair2score:
            cand.append((key, pair2score[key]))
    cand.sort(key=lambda kv: (-kv[1], kv[0]))

    dsu = DSU(tables_in_query)
    chosen: List[Tuple[str, str]] = []
    connected: Set[str] = set()

    for (a, b), sc in cand:
        if dsu.union(a, b):
            chosen.append((a, b))
            connected.add(a); connected.add(b)

    if not chosen:
        out = {"mode": "fallback_rowplan", "columnar_set": sorted(CS), "join_order_edges": row_edges}
        Path(cfg["OUT_JSON"]).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(cfg["OUT_TXT"], "w", encoding="utf-8") as f:
            f.write("# Join order (fallback to row plan)\n")
            for i, (u, v) in enumerate(row_edges, 1):
                f.write(f"{i:>2}. {u} ⋈ {v}\n")
        print(f"CS is empty; fallback written to: {cfg['OUT_JSON']} / {cfg['OUT_TXT']}")
        return

    remaining = sorted(list(tables_in_query - connected), key=lambda t: -row_depth.get(t, 10**6))

    def best_priority_neighbor(t: str, comp: Set[str]) -> Optional[str]:
        best, best_sc = None, float("-inf")
        for x in comp:
            sc = pair2score.get(tuple(sorted((t, x))))
            if sc is None: 
                continue
            if sc > best_sc:
                best_sc, best = sc, x
        return best

    def first_row_neighbor_in_comp(t: str, comp: Set[str]) -> Optional[str]:
        for nb in row_neighbors.get(t, []):
            if nb in comp:
                return nb
        return None

    for t in remaining:
        x = best_priority_neighbor(t, connected)
        if x is not None and dsu.union(t, x):
            chosen.append(tuple(sorted((t, x))))
            connected.add(t)
            continue
        y = first_row_neighbor_in_comp(t, connected)
        if y is not None and dsu.union(t, y):
            chosen.append(tuple(sorted((t, y))))
            connected.add(t)
            continue
        if connected:
            z = next(iter(connected))
            if dsu.union(t, z):
                chosen.append(tuple(sorted((t, z))))
                connected.add(t)

    out = {
        "mode": "hybrid_greedy",
        "columnar_set": sorted(CS),
        "join_order_edges": chosen,
        "row_spanning_tree": row_edges,
        "row_leaf_depth": row_depth,
    }
    Path(cfg["OUT_JSON"]).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(cfg["OUT_TXT"], "w", encoding="utf-8") as f:
        f.write("# Join order (hybrid greedy + row-depth attach)\n")
        for i, (u, v) in enumerate(chosen, 1):
            f.write(f"{i:>2}. {u} ⋈ {v}\n")
    print(f"Done: {cfg['OUT_JSON']} / {cfg['OUT_TXT']}")

if __name__ == "__main__":
    main()

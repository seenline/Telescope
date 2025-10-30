#!/usr/bin/env python3

from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ★ 直接 import 工具模块
from simplify_plan_json import simplify_plan_json
from plan_traversal import get_traversals_from_plan_json

CONFIG = {
    "ENUM_PLAN_PATH": "enum_plan.txt",
    "OUT_JSON": "pair_priority.json",
    "OUT_TXT": "pair_priority.txt",
    "PAIR_HEADER_PATTERN": r"^#\s*Pair\s+\d+/\d+:\s*\(\s*([^.()\s,]+)\s*,\s*([^.()\s,]+)\s*\)\s*$",
    "QUERY_HEADER_PATTERN": r"^--\s*Query\s+\d+",
}
BANNER = "#" * 80

Pair = Tuple[str, str]

@dataclass
class OneQueryPlan:
    pair: Pair
    query_idx: int
    raw_json: str

def _extract_json_block(text: str) -> Optional[str]:
    start = None
    stack = []
    for i, ch in enumerate(text):
        if ch in "{[":
            if start is None:
                start = i
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                continue
            need = stack.pop()
            if (ch == "}" and need != "}") or (ch == "]" and need != "]"):
                pass
            if start is not None and not stack:
                return text[start : i + 1]
    return None

def parse_enum_plan_file(path: str, cfg=CONFIG) -> List[OneQueryPlan]:
    pair_re = re.compile(cfg["PAIR_HEADER_PATTERN"])
    q_re = re.compile(cfg["QUERY_HEADER_PATTERN"])
    plans: List[OneQueryPlan] = []
    cur_pair: Optional[Pair] = None
    cur_q: Optional[int] = None
    buf: List[str] = []

    def flush():
        nonlocal cur_pair, cur_q, buf, plans
        if cur_pair is None or cur_q is None:
            buf = []
            return
        block = "".join(buf).strip()
        buf = []
        if not block:
            return
        json_str = _extract_json_block(block)
        if json_str:
            plans.append(OneQueryPlan(cur_pair, cur_q, json_str))

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            m = pair_re.match(s)
            if m:
                flush()
                cur_pair = (m.group(1), m.group(2))
                cur_q = None
                continue
            if q_re.match(s):
                flush()
                m2 = re.search(r"Query\s+(\d+)", s)
                cur_q = int(m2.group(1)) if m2 else (0 if cur_q is None else cur_q + 1)
                continue
            buf.append(line)
    flush()
    return plans

def _children(n: Dict[str, Any]) -> List[Dict[str, Any]]:
    ch = n.get("Plans") or []
    return [c for c in ch if isinstance(c, dict)] if isinstance(ch, list) else []

def _node_type(n: Dict[str, Any]) -> str:
    return str(n.get("Node Type", "")).lower()

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

def _find_all_leaves(root: Dict[str, Any]) -> List[Dict[str, Any]]:
    out, st = [], [root]
    while st:
        n = st.pop()
        ch = _children(n)
        if not ch:
            out.append(n)
        else:
            st.extend(reversed(ch))
    return out

def _find_leaf(root: Dict[str, Any], table: str) -> Optional[Dict[str, Any]]:
    tgt = _normalize_tbl(table)
    for lf in _find_all_leaves(root):
        lab = _leaf_label(lf)
        if lab and _normalize_tbl(lab) == tgt:
            return lf
    return None

def _parent_map(root: Dict[str, Any]) -> Dict[int, Optional[int]]:
    p = {id(root): None}
    st = [root]
    while st:
        n = st.pop()
        for c in _children(n):
            p[id(c)] = id(n)
            st.append(c)
    return p

def _depth_of(root: Dict[str, Any], node: Dict[str, Any]) -> int:
    p = _parent_map(root)
    d, cur = 0, id(node)
    while p.get(cur) is not None:
        cur = p[cur]  # type: ignore
        d += 1
    return d

def _lca(root: Dict[str, Any], a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    p = _parent_map(root)
    seen = set()
    cur = id(a)
    while True:
        seen.add(cur)
        nxt = p.get(cur)
        if nxt is None:
            break
        cur = nxt
    cur = id(b)
    while cur not in seen:
        nxt = p.get(cur)
        if nxt is None:
            break
        cur = nxt
    target = cur
    st = [root]
    while st:
        n = st.pop()
        if id(n) == target:
            return n
        st.extend(_children(n))
    return root

@dataclass
class PairAgg:
    depths: List[int]
    miss: int
    preorder: List[str]
    inorder: List[str]
    score: float = 0.0
    avg_depth: Optional[float] = None
    hit_ratio: float = 0.0

def main():
    cfg = CONFIG
    items = parse_enum_plan_file(cfg["ENUM_PLAN_PATH"], cfg)

    agg: Dict[Pair, PairAgg] = {}

    for it in items:
        try:
            simplified_str = simplify_plan_json(it.raw_json)
            simp_obj = json.loads(simplified_str)
            root = _root_from_simplified(simp_obj)
        except Exception:
            continue

        preorder, inorder = get_traversals_from_plan_json(root)

        a, b = it.pair
        la, lb = _find_leaf(root, a), _find_leaf(root, b)

        if it.pair not in agg:
            agg[it.pair] = PairAgg(depths=[], miss=0, preorder=preorder, inorder=inorder)

        if la is None or lb is None:
            agg[it.pair].miss += 1
            continue

        lca = _lca(root, la, lb)
        d = _depth_of(root, lca)
        agg[it.pair].depths.append(d)

    results = []
    for pair, a in agg.items():
        total = len(a.depths) + a.miss
        hit = len(a.depths)
        a.hit_ratio = (hit / total) if total else 0.0
        a.avg_depth = (sum(a.depths) / hit) if hit else None
        a.score = a.hit_ratio * (1.0 / (1.0 + (a.avg_depth if a.avg_depth is not None else 1e9)))
        results.append((pair, a))

    results.sort(key=lambda r: (-r[1].score, r[1].avg_depth if r[1].avg_depth is not None else 1e9, -r[1].hit_ratio, r[0]))

    Path(cfg["OUT_JSON"]).write_text(
        json.dumps(
            [
                {
                    "pair": list(pair),
                    "score": a.score,
                    "avg_depth": a.avg_depth,
                    "hit_ratio": a.hit_ratio,
                    "depths": a.depths,
                    "preorder_example": a.preorder,
                    "inorder_example": a.inorder,
                    "miss": a.miss,
                }
                for pair, a in results
            ],
            ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )

    with open(cfg["OUT_TXT"], "w", encoding="utf-8") as f:
        f.write("# Pair Priority Ranking\n\n")
        for i, (pair, a) in enumerate(results, 1):
            f.write(f"{i:>3}. ({pair[0]}, {pair[1]})  score={a.score:.6f}  "
                    f"avg_depth={a.avg_depth if a.avg_depth is not None else 'NA'}  "
                    f"hit_ratio={a.hit_ratio:.2%}  miss={a.miss}\n")
    print(f"Done: {cfg['OUT_JSON']} / {cfg['OUT_TXT']}")

if __name__ == "__main__":
    main()

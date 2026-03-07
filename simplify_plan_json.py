import json
from typing import Any, Dict, List, Optional


def _is_join_node(node: Dict[str, Any]) -> bool:
    node_type = node.get("Node Type", "") or ""
    if not isinstance(node_type, str):
        return False
    if node_type == "Nested Loop":
        return True
    return "Join" in node_type


def _is_scan_node(node: Dict[str, Any]) -> bool:
    node_type = node.get("Node Type", "") or ""
    if not isinstance(node_type, str):
        return False
    return "Scan" in node_type


def _append_has_custom_scan(node: Dict[str, Any]) -> bool:
    if node.get("Node Type") != "Append":
        return False
    for child in node.get("Plans", []) or []:
        if child.get("Node Type") == "Custom Scan":
            return True
    return False


def _pick_first_custom_scan_child(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for child in node.get("Plans", []) or []:
        if child.get("Node Type") == "Custom Scan":
            return child
    return None


def _copy_if_present(src: Dict[str, Any], dst: Dict[str, Any], keys: List[str]) -> None:
    for key in keys:
        if key in src:
            dst[key] = src[key]


def _simplify_node(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(node, dict):
        return None

    node_type = node.get("Node Type", "")

    if node_type == "Append" and _append_has_custom_scan(node):
        custom_child = _pick_first_custom_scan_child(node)
        simplified: Dict[str, Any] = {"Node Type": "Custom Scan"}
        if custom_child is not None:
            _copy_if_present(custom_child, simplified, [
                "Relation Name",
                "Alias",
            ])
        _copy_if_present(node, simplified, [
            "Plan Rows",
            "Total Cost",
        ])
        return simplified

    if _is_join_node(node):
        simplified_join: Dict[str, Any] = {
            "Node Type": node.get("Node Type"),
        }
        _copy_if_present(node, simplified_join, [
            "Join Type",
            "Plan Rows",
            "Total Cost",
        ])

        children: List[Dict[str, Any]] = []
        for child in node.get("Plans", []) or []:
            c = _simplify_node(child)
            if c is not None:
                children.append(c)
        if children:
            simplified_join["Plans"] = children
        return simplified_join

    if _is_scan_node(node):
        simplified_scan: Dict[str, Any] = {
            "Node Type": node.get("Node Type"),
        }
        _copy_if_present(node, simplified_scan, [
            "Plan Rows",
            "Total Cost",
        ])
        _copy_if_present(node, simplified_scan, [
            "Relation Name",
            "Alias",
        ])
        return simplified_scan

    simplified_children: List[Dict[str, Any]] = []
    for child in node.get("Plans", []) or []:
        c = _simplify_node(child)
        if c is not None:
            simplified_children.append(c)

    if not simplified_children:
        return None

    return simplified_children[0]


def _simplify_plan_root(plan_obj: Any) -> Any:
    if isinstance(plan_obj, list):
        out: List[Any] = []
        for entry in plan_obj:
            if isinstance(entry, dict) and "Plan" in entry:
                simplified_root = _simplify_node(entry["Plan"]) or {}
                out.append({"Plan": simplified_root})
            else:
                simplified_root = _simplify_node(entry) or {}
                out.append(simplified_root)
        return out

    if isinstance(plan_obj, dict) and "Plan" in plan_obj:
        return {"Plan": _simplify_node(plan_obj["Plan"]) or {}}

    if isinstance(plan_obj, dict):
        return _simplify_node(plan_obj) or {}

    return {}


def simplify_plan_json(plan_json_str: str) -> str:
    data = json.loads(plan_json_str)
    simplified = _simplify_plan_root(data)
    return json.dumps(simplified, ensure_ascii=False)


__all__ = ["simplify_plan_json"]



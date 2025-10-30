 

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def _label_node(node: Dict) -> str:
    

    node_type: str = str(node.get("Node Type", "")).strip()
    lower_node_type = node_type.lower()

    if "scan" in lower_node_type:
        relation_name = node.get("Relation Name")
        if isinstance(relation_name, str) and relation_name:
            return relation_name
        alias_name = node.get("Alias")
        if isinstance(alias_name, str) and alias_name:
            return alias_name
        return node_type

    if ("join" in lower_node_type) or ("nested loop" in lower_node_type):
        return "join"

    return node_type


def _ordered_children(node: Dict) -> List[Dict]:
    

    children: List[Dict] = list(node.get("Plans") or [])

    def priority(child: Dict) -> Tuple[int, int]:
        rel = str(child.get("Parent Relationship", "")).strip().lower()
        if rel == "outer":
            return (0, 0)
        if rel == "inner":
            return (1, 0)
        if rel == "member":
            return (2, 0)
        return (3, 0)

    return sorted(children, key=priority)


def _preorder(node: Dict, output: List[str]) -> None:
    output.append(_label_node(node))
    for child in _ordered_children(node):
        _preorder(child, output)


def _inorder(node: Dict, output: List[str]) -> None:
    children = _ordered_children(node)
    num_children = len(children)
    if num_children == 0:
        output.append(_label_node(node))
        return

    if num_children == 1:
        _inorder(children[0], output)
        output.append(_label_node(node))
        return

    _inorder(children[0], output)
    output.append(_label_node(node))
    for idx in range(1, num_children):
        _inorder(children[idx], output)


def get_traversals_from_plan_root(plan_root: Dict) -> Tuple[List[str], List[str]]:
    

    preorder: List[str] = []
    inorder: List[str] = []
    _preorder(plan_root, preorder)
    _inorder(plan_root, inorder)
    return preorder, inorder


def get_traversals_from_plan_json(plan_json) -> Tuple[List[str], List[str]]:
    

    root = None
    if isinstance(plan_json, dict):
        root = plan_json.get("Plan", plan_json)
    elif isinstance(plan_json, list):
        for element in plan_json:
            if isinstance(element, dict) and ("Plan" in element or "Node Type" in element):
                root = element.get("Plan", element)
                break
        if root is None and plan_json:
            first = plan_json[0]
            if isinstance(first, dict):
                root = first.get("Plan", first)
    else:
        raise TypeError("Unsupported plan_json type: {}".format(type(plan_json)))

    if root is None:
        raise ValueError("Could not determine plan root from provided JSON structure")

    return get_traversals_from_plan_root(root)


def get_traversals_from_file(file_path: str) -> Tuple[List[str], List[str]]:
    
    import json
    from json import JSONDecodeError

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text_stripped = text.lstrip()
    try:
        data = json.loads(text_stripped)
    except JSONDecodeError:
        decoder = json.JSONDecoder()
        try:
            data, _ = decoder.raw_decode(text_stripped)
        except JSONDecodeError:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    break
                except JSONDecodeError:
                    continue
            else:
                raise

    return get_traversals_from_plan_json(data)


__all__ = [
    "get_traversals_from_plan_root",
    "get_traversals_from_plan_json",
    "get_traversals_from_file",
]



#!/usr/bin/env python3
"""
Static reachability + reference analysis to flag unused files and code.
- Walks the repo (excluding ICICLE-Benchmark and common binary/cache dirs)
- Builds import graph
- Finds roots by grepping for entrypoints (main.py) and src package
- Reports files never imported/referenced
- Inside Python files: flags unused top-level functions, classes, and constants
Limitations: dynamic imports, runtime dispatch, CLI scripts, plugins may be missed.
"""
from __future__ import annotations
import ast
import os
import re
import sys
from pathlib import Path
from collections import defaultdict, deque

REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {
    "ICICLE-Benchmark",  # user requested exclusion
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "logs",
    "pretrained_weight",
    "data",
}

PY_EXTS = {".py"}

class ModuleInfo:
    def __init__(self, path: Path, modname: str):
        self.path = path
        self.modname = modname
        self.imports: set[str] = set()
        self.defs: dict[str, tuple[str, int]] = {}  # name -> (kind, lineno)
        self.refs: set[str] = set()


def rel_module_name(path: Path) -> str | None:
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        return None
    if any(part in EXCLUDE_DIRS for part in rel.parts[:-1]):
        return None
    if path.suffix not in PY_EXTS:
        return None
    # convert path to module like src/foo/bar.py -> src.foo.bar
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def iter_python_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def parse_module(path: Path) -> ModuleInfo | None:
    modname = rel_module_name(path)
    if not modname:
        return None
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return None

    info = ModuleInfo(path, modname)

    class Analyzer(ast.NodeVisitor):
        def __init__(self):
            self.class_depth = 0
            super().__init__()
        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                info.imports.add(alias.name.split(".")[0])

        def visit_ImportFrom(self, node: ast.ImportFrom):
            if node.module:
                base = node.module.split(".")[0]
                info.imports.add(base)

        def visit_FunctionDef(self, node: ast.FunctionDef):
            if isinstance(getattr(node, "decorator_list", []), list) and any(
                getattr(d, "id", None) == "property" for d in node.decorator_list
            ):
                kind = "property"
            else:
                kind = "method" if self.class_depth > 0 else "function"
            info.defs[node.name] = (kind, node.lineno)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef):
            info.defs[node.name] = ("class", node.lineno)
            self.class_depth += 1
            self.generic_visit(node)
            self.class_depth -= 1

        def visit_Assign(self, node: ast.Assign):
            # capture top-level constants (UPPER_CASE)
            if isinstance(getattr(node, "targets", None), list):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id.isupper():
                        info.defs[t.id] = ("const", node.lineno)
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name):
            info.refs.add(node.id)

        def visit_Attribute(self, node: ast.Attribute):
            # Capture attribute names, e.g., obj.method -> 'method'
            if isinstance(node.attr, str):
                info.refs.add(node.attr)
            self.generic_visit(node)

    Analyzer().visit(tree)
    return info


def build_graph() -> dict[str, ModuleInfo]:
    modules: dict[str, ModuleInfo] = {}
    for path in iter_python_files(REPO_ROOT):
        mi = parse_module(path)
        if mi:
            modules[mi.modname] = mi
    return modules


def find_roots(mods: dict[str, ModuleInfo]) -> set[str]:
    roots = set()
    # Common entrypoints: main.py, src package, scripts/*.py
    for m in mods.values():
        rel = m.path.relative_to(REPO_ROOT).as_posix()
        if rel == "main.py" or rel.startswith("scripts/") or rel.startswith("src/"):
            roots.add(m.modname)
    return roots


def reachable_modules(mods: dict[str, ModuleInfo], roots: set[str]) -> set[str]:
    reachable = set()
    q = deque(roots)
    while q:
        cur = q.popleft()
        if cur in reachable or cur not in mods:
            continue
        reachable.add(cur)
        # approximate: if an import matches a top-level package/module name within repo, enqueue
        cur_dir = mods[cur].path.parent
        for imp in mods[cur].imports:
            # find modules whose top-level package equals imp
            for name, mi in mods.items():
                top = name.split(".")[0]
                if top == imp and name not in reachable:
                    q.append(name)
    return reachable


def find_unused_defs(mods: dict[str, ModuleInfo]) -> dict[str, list[tuple[str, str, int]]]:
    # returns modname -> list of (name, kind, lineno)
    # If a def name is never referenced anywhere across reachable set, mark unused
    all_refs = set()
    for mi in mods.values():
        all_refs |= mi.refs
    unused: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for name, mi in mods.items():
        for defname, (kind, lineno) in mi.defs.items():
            # exclude dunder, main guards, typing aliases
            if defname.startswith("__") and defname.endswith("__"):
                continue
            if defname in {"main", "__all__"}:
                continue
            if defname not in all_refs:
                unused[name].append((defname, kind, lineno))
    return unused


def main():
    mods = build_graph()
    roots = find_roots(mods)
    reach = reachable_modules(mods, roots)

    never_imported = sorted([m for m in mods.keys() if m not in reach])

    unused_defs = find_unused_defs({m: mods[m] for m in reach})

    print("Repo root:", REPO_ROOT)
    print("Total modules:", len(mods))
    print("Roots ({}):".format(len(roots)))
    for r in sorted(roots):
        print("  -", r)
    print()
    print("Modules not reachable from roots (potentially unused): {}".format(len(never_imported)))
    for m in never_imported:
        print("  -", m, "->", mods[m].path.relative_to(REPO_ROOT))

    print()
    print("Potentially unused definitions inside reachable modules:")
    for m in sorted(unused_defs.keys()):
        if not unused_defs[m]:
            continue
        print("-", m, "->", mods[m].path.relative_to(REPO_ROOT))
        for name, kind, lineno in sorted(unused_defs[m], key=lambda x: x[2]):
            print(f"    L{lineno:<4} {kind:<8} {name}")

    print()
    print("Notes:")
    print("- This is static and conservative. Dynamic imports and CLI entrypoints may be missed.")
    print("- Review before deleting. You can whitelist files by adding to EXCLUDE_DIRS.")

if __name__ == "__main__":
    main()

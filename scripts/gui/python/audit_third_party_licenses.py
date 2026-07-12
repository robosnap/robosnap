#!/usr/bin/env python3
"""Audit visible license files for dependencies listed in third_party/manifest.yaml."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


GUI_COMPONENTS = {"sam3", "sam-3d-objects", "Hunyuan3D-Part/P3-SAM"}
LICENSE_NAMES = {"license", "license.txt", "license.md", "copying", "notice", "authors"}


def parse_scalar(value: str) -> str:
    value = value.strip()
    if value[:1] in {"'", '"'}:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value.strip("'\"")
        return str(parsed)
    return value


def read_components(path: Path) -> list[dict[str, str]]:
    components: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("  - name:"):
            if current:
                components.append(current)
            current = {"name": parse_scalar(line.split(":", 1)[1])}
        elif current is not None and line.startswith("    ") and ":" in line:
            key, value = line.strip().split(":", 1)
            current[key] = parse_scalar(value)
    if current:
        components.append(current)
    return components


def visible_license_files(path: Path, component: dict[str, str], root: Path) -> list[Path]:
    explicit = []
    for key in ("license_file", "notice_file"):
        value = component.get(key)
        if value:
            candidate = root / value
            if candidate.is_file():
                explicit.append(candidate)
    if explicit:
        return explicit
    if not path.is_dir():
        return []
    return sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.name.lower() in LICENSE_NAMES
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-components", action="store_true")
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    manifest = root / "third_party" / "manifest.yaml"
    components = read_components(manifest)
    if not args.all_components:
        components = [item for item in components if item["name"] in GUI_COMPONENTS]

    failures = 0
    for component in components:
        release_path = component.get("release_path")
        if not release_path:
            print(f"[MISSING_PATH] {component['name']}")
            failures += 1
            continue
        path = root / release_path
        if not release_path.startswith("third_party/"):
            print(f"[FIRST_PARTY] {component['name']}: {release_path}")
            continue
        if not path.exists():
            print(f"[MISSING_SOURCE] {component['name']}: {release_path}")
            failures += 1
            continue
        licenses = visible_license_files(path, component, root)
        if not licenses:
            print(f"[MISSING_LICENSE] {component['name']}: {release_path}")
            failures += 1
            continue
        shown = ", ".join(str(item.relative_to(root)) for item in licenses)
        print(f"[OK] {component['name']}: {shown}")

    print(f"components={len(components)} failures={failures}")
    return 1 if args.fail_on_missing and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

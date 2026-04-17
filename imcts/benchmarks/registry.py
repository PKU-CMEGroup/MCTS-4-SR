"""Bundled benchmark registry and case catalog access."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any


@dataclass(frozen=True)
class BenchmarkGroup:
    name: str
    source_type: str
    default_config_name: str
    needs_constant_op: bool
    case_catalog_key: str


class CaseCatalog:
    def __init__(self, cases_by_group: dict[str, list[dict[str, Any]]]):
        self._cases_by_group = {group: list(cases) for group, cases in cases_by_group.items()}

    def list_groups(self) -> list[str]:
        return list(self._cases_by_group)

    def get_cases(self, group: str) -> list[dict[str, Any]]:
        if group not in self._cases_by_group:
            available = ", ".join(self.list_groups())
            raise ValueError(f"Unknown group '{group}'. Available groups: {available}")
        return list(self._cases_by_group[group])


class BenchmarkRegistry:
    def __init__(self, groups: list[BenchmarkGroup], catalog: CaseCatalog):
        self._groups = {group.name: group for group in groups}
        self._catalog = catalog

    def list_groups(self) -> list[str]:
        return list(self._groups)

    def get_group(self, name: str) -> BenchmarkGroup:
        if name not in self._groups:
            available = ", ".join(self.list_groups())
            raise ValueError(f"Unknown group '{name}'. Available groups: {available}")
        return self._groups[name]

    def get_cases(self, group_name: str) -> list[dict[str, Any]]:
        group = self.get_group(group_name)
        return self._catalog.get_cases(group.case_catalog_key)

    def available_groups(self) -> dict[str, list[dict[str, Any]]]:
        return {group_name: self.get_cases(group_name) for group_name in self.list_groups()}

    def select_cases(self, group_name: str, selection: str) -> list[dict[str, Any]]:
        """Resolve ``--cases`` syntax into the concrete case list.

        Supported forms are:

        - ``all``
        - comma-separated numeric ids, such as ``1,2,4``
        - numeric ranges, such as ``1-4``
        - exact case names, matched case-insensitively
        """
        cases = self.get_cases(group_name)
        if selection.strip().lower() == "all":
            return cases

        wanted_ids: set[int] = set()
        wanted_names: set[str] = set()
        for token in (part.strip() for part in selection.split(",") if part.strip()):
            if "-" in token and token.replace("-", "").isdigit():
                start, end = (int(x) for x in token.split("-", 1))
                wanted_ids.update(range(min(start, end), max(start, end) + 1))
            elif token.isdigit():
                wanted_ids.add(int(token))
            else:
                wanted_names.add(token.lower())

        selected = [
            case
            for case in cases
            if case["id"] in wanted_ids or case["name"].lower() in wanted_names
        ]
        if not selected:
            raise ValueError(f"No cases matched --cases={selection!r} in group {group_name!r}.")
        return selected


def load_json_resource(name: str) -> dict[str, Any]:
    with resources.files("imcts.benchmarks").joinpath(name).open("r", encoding="utf-8") as f:
        return json.load(f)


def print_available_cases(registry: BenchmarkRegistry, selected_group: str | None = None) -> None:
    for group_name in registry.list_groups():
        if selected_group is not None and group_name != selected_group:
            continue
        print(f"{group_name}:")
        for case in registry.get_cases(group_name):
            expression = case.get("expression")
            if expression:
                print(f"  {case['id']:>3}: {case['name']}  y = {expression}")
            else:
                print(f"  {case['id']:>3}: {case['name']}")


def load_bundled_registry() -> BenchmarkRegistry:
    basic = load_json_resource("basic.json")
    blackbox = load_json_resource("blackbox.json")

    symbolic_groups = basic["groups"]
    requires_constants = set(basic.get("constant_groups", {}).get("requires_constants", []))
    no_constants = set(basic.get("constant_groups", {}).get("no_constants", []))
    overlap = requires_constants & no_constants
    if overlap:
        formatted = ", ".join(sorted(overlap))
        raise ValueError(f"Groups cannot both require and forbid constants: {formatted}")

    unknown_groups = (requires_constants | no_constants) - set(symbolic_groups)
    if unknown_groups:
        formatted = ", ".join(sorted(unknown_groups))
        raise ValueError(f"Constant-group metadata references unknown groups: {formatted}")

    cases_by_group: dict[str, list[dict[str, Any]]] = {
        group_name: list(cases)
        for group_name, cases in symbolic_groups.items()
    }
    cases_by_group["BlackBox"] = [
        {"id": idx, "name": name}
        for idx, name in enumerate(blackbox["BlackBox"], start=1)
    ]
    catalog = CaseCatalog(cases_by_group)

    groups = [
        BenchmarkGroup(
            name=group_name,
            source_type="expression",
            default_config_name="basic.yaml",
            needs_constant_op=group_name in requires_constants,
            case_catalog_key=group_name,
        )
        for group_name in symbolic_groups
    ]
    groups.append(
        BenchmarkGroup(
            name="BlackBox",
            source_type="dataset",
            default_config_name="blackbox.yaml",
            needs_constant_op=True,
            case_catalog_key="BlackBox",
        )
    )
    return BenchmarkRegistry(groups=groups, catalog=catalog)

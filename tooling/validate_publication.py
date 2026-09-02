"""Fail CI when release or JOSS-publication invariants are broken."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "paper.md"
BIBLIOGRAPHY = ROOT / "paper" / "paper.bib"


def main() -> int:
    failures: list[str] = []
    required_files = (
        ROOT / "LICENSE",
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "CODE_OF_CONDUCT.md",
        ROOT / "GOVERNANCE.md",
        ROOT / "SECURITY.md",
        ROOT / "CITATION.cff",
        ROOT / "CHANGELOG.md",
        ROOT / "JOSS_READINESS.md",
        PAPER,
        BIBLIOGRAPHY,
        ROOT / "openprism" / "spec" / "ai-scene-digest.schema.json",
        ROOT / "openprism" / "policy" / "fusion_policy_v1.json",
    )
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty required file: {path.relative_to(ROOT)}")

    if PAPER.is_file():
        source = PAPER.read_text(encoding="utf-8")
        body = source.split("---", 2)[-1]
        body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
        words = re.findall(r"\b[\w'-]+\b", body)
        if not 750 <= len(words) <= 1750:
            failures.append(f"JOSS paper body has {len(words)} words; expected 750-1750")
        required_headings = (
            "# Summary",
            "# Statement of need",
            "# State of the field",
            "# Software design",
            "# Research impact statement",
            "# AI usage disclosure",
            "# Acknowledgements",
            "# References",
        )
        for heading in required_headings:
            if heading not in source:
                failures.append(f"paper is missing required heading: {heading}")
        citation_keys = set(re.findall(r"@([A-Za-z0-9:_-]+)", source))
        bib_keys = set(re.findall(r"@[A-Za-z]+\{\s*([^,\s]+)", BIBLIOGRAPHY.read_text(encoding="utf-8")))
        missing_keys = sorted(citation_keys - bib_keys)
        if missing_keys:
            failures.append(f"paper cites missing bibliography keys: {missing_keys}")

    for relative in (
        "openprism/spec/ai-scene-digest.schema.json",
        "openprism/spec/prism-frame.schema.json",
        "openprism/policy/fusion_policy_v1.json",
    ):
        path = ROOT / relative
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"invalid JSON in {relative}: {error}")

    if failures:
        print("Publication validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Publication artifacts satisfy the automated preflight checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

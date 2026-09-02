"""Validate the PRISM-EGT TMLR draft and fail closed for submission mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT / "paper" / "tmlr"


def validate(*, submission_ready: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        "paper.tex",
        "references.bib",
        "tmlr.sty",
        "tmlr.bst",
        "fancyhdr.sty",
        "TMLR_STYLE_LICENSE",
        "EXPERIMENT_PROTOCOL.md",
        "baselines.lock.json",
        "NOVELTY_AUDIT.md",
        "SUBMISSION_CHECKLIST.md",
    )
    for filename in required:
        if not (PAPER_ROOT / filename).is_file():
            errors.append(f"missing paper/tmlr/{filename}")
    if errors:
        return errors

    tex = (PAPER_ROOT / "paper.tex").read_text(encoding="utf-8")
    bib = (PAPER_ROOT / "references.bib").read_text(encoding="utf-8")
    try:
        baseline_lock = json.loads(
            (PAPER_ROOT / "baselines.lock.json").read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as error:
        errors.append(f"baseline lock is not readable JSON: {error}")
        baseline_lock = {}
    candidates = baseline_lock.get("candidates", [])
    if len(candidates) < 3:
        errors.append("baseline lock must contain at least three learned candidates")
    for candidate in candidates:
        missing_fields = {
            "id", "repository", "revision", "repository_license", "execution_status"
        } - set(candidate)
        if missing_fields:
            errors.append(
                f"baseline candidate is missing fields: {sorted(missing_fields)}"
            )
    if "\\usepackage{tmlr}" not in tex:
        errors.append("paper does not use the anonymous TMLR style")
    if "\\usepackage[accepted]{tmlr}" in tex or "\\usepackage[preprint]{tmlr}" in tex:
        errors.append("review draft must use the anonymous TMLR style")
    if "Anonymous Author" not in tex:
        errors.append("review draft does not have an anonymous author block")
    if "large language model" not in tex.lower():
        errors.append("first-page AI-assistance disclosure is missing")
    if "\\bibliographystyle{tmlr}" not in tex:
        errors.append("paper does not use the TMLR bibliography style")

    citations: set[str] = set()
    for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", tex):
        citations.update(item.strip() for item in group.split(","))
    references = set(re.findall(r"@[A-Za-z]+\{\s*([^,\s]+)", bib))
    missing = sorted(citations - references)
    if missing:
        errors.append(f"citation keys missing from bibliography: {missing}")

    # A cheap structural check catches common incomplete edits when a TeX
    # executable is unavailable. It does not replace a real PDF build.
    if tex.count("{") != tex.count("}"):
        errors.append("paper.tex has unbalanced braces")
    if tex.count("\\begin{") != tex.count("\\end{"):
        errors.append("paper.tex has unmatched begin/end environments")

    if submission_ready:
        blockers = []
        for token in ("TBD", "implementation-complete protocol draft"):
            if token.lower() in tex.lower():
                blockers.append(token)
        checklist = (PAPER_ROOT / "SUBMISSION_CHECKLIST.md").read_text(
            encoding="utf-8"
        )
        if "- [ ]" in checklist:
            blockers.append("unchecked submission checklist items")
        if not (PAPER_ROOT / "paper.pdf").is_file():
            blockers.append("rendered paper.pdf")
        if not baseline_lock.get("policy", {}).get("final_selection_frozen", False):
            blockers.append("external baseline selection and artifacts are not frozen")
        if any(
            candidate.get("execution_status") != "complete"
            for candidate in candidates
        ):
            blockers.append("one or more external learned baselines have not executed")
        if blockers:
            errors.append(
                "submission is intentionally blocked by: " + ", ".join(blockers)
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-ready", action="store_true")
    args = parser.parse_args()
    errors = validate(submission_ready=args.submission_ready)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(
        "TMLR source draft is structurally valid."
        + (" Submission gates passed." if args.submission_ready else " Scientific submission gates remain separate.")
    )


if __name__ == "__main__":
    main()

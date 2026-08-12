#!/usr/bin/env python3
"""Refresh the "Latest Results" section of README.md.

Regenerates the block between the markers
``<!-- latest-results:start -->`` and ``<!-- latest-results:end -->``
from the detection metrics files (reports/metrics_*.json) and the split
object counts (reports/class_distribution.json). If README.md does not
exist it is created with a minimal skeleton containing the marker pair.

Pure stdlib — safe to run in CI:
  python scripts/update_readme_results.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

START = "<!-- latest-results:start -->"
END = "<!-- latest-results:end -->"

NOTE = (
    "> Note: variant-22 mAP is not directly comparable to variant-11 mAP "
    "(different class sets). It is included as reference only; the real "
    "Approach A vs B comparison happens after the classifier stage."
)

README_SKELETON = f"""# War Damage Assessment

War damage assessment via YOLO detection + damage classifiers, built on a
DVC + MLflow + CML MLOps stack.

## Latest Results

{START}
{END}
"""


def load_json(path: Path) -> dict | None:
    """Parse a JSON file, returning None on any read/parse failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def metrics_table(reports_dir: Path) -> list[str]:
    """Markdown table of detection metrics, or a placeholder if none exist."""
    rows = []
    for path in sorted(reports_dir.glob("metrics_*.json")):
        m = load_json(path)
        if m is not None:
            rows.append(m)
    if not rows:
        return ["_No results yet — run dvc repro._", "", NOTE]
    lines = [
        "| model | classes | mAP50 | mAP50-95 | precision | recall |",
        "|---|---|---|---|---|---|",
    ]
    for m in rows:
        cells = [str(m.get(k, "-")) for k in
                 ("model", "variant", "mAP50", "mAP50_95", "precision", "recall")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", NOTE]
    return lines


def split_counts(reports_dir: Path) -> list[str]:
    """Per-split total object counts (11-class scheme), if the report exists."""
    dist = load_json(reports_dir / "class_distribution.json")
    if not dist or "scheme11" not in dist:
        return []
    scheme11 = dist["scheme11"]
    lines = ["", "**Objects per split (11-class scheme):**", ""]
    for split in ("train", "val", "test"):
        total = sum(scheme11.get(split, {}).values())
        lines.append(f"- {split}: {total}")
    return lines


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    readme = root / "README.md"
    reports_dir = root / "reports"

    if not readme.is_file():
        readme.write_text(README_SKELETON, encoding="utf-8")
        print(f"created {readme}")

    text = readme.read_text(encoding="utf-8")
    if START not in text or text.find(END, text.find(START)) == -1:
        print("markers not found — appending a fresh Latest Results section")
        text = text.rstrip("\n") + f"\n\n## Latest Results\n\n{START}\n{END}\n"

    section = ["", f"_Generated: {datetime.now().isoformat(timespec='minutes')}_", ""]
    section += metrics_table(reports_dir)
    section += split_counts(reports_dir)
    section.append("")

    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    readme.write_text(head + START + "\n".join(section) + "\n" + END + tail,
                      encoding="utf-8")
    print(f"updated {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

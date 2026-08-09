#!/usr/bin/env python3
"""Phase 0 data gate for War-Damage-Assessment.

Validates YOLO+damage-flag label files (expected: 6 columns per line) and,
optionally, compares your LOCAL authoritative labels against labels extracted
from a git branch (e.g. feature/startannotation) to prove nothing is lost
before that branch is deleted.

Usage (run in Git Bash / any terminal with Python 3.10+):

  # 1) Validate local labels only
  python verify_labels.py --labels "path/to/data/annotations/labels"

  # 2) Validate AND compare against branch-extracted labels + write report
  python verify_labels.py --labels "path/to/local/labels" \
      --compare "path/to/branch_annotations/data/annotations/labels" \
      --report gate_report.md

  # 3) Also check image<->label correspondence (do this for the final dataset,
  #    especially since new images were added)
  python verify_labels.py --labels "path/to/local/labels" \
      --images "path/to/local/data/raw" --report gate_report.md

  # 4) After review: normalize line endings (CRLF -> LF) in-place
  python verify_labels.py --labels "path/to/local/labels" --fix-eol

Exit codes (bit flags, may combine):
  0 = PASS (safe to proceed)
  1 = validation errors in local labels (fix before dvc add)
  2 = files exist on branch but are MISSING locally (data-loss risk - do NOT
      delete the branch; copy the missing files first)
  4 = orphan labels: label files whose image does not exist
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_NUM_CLASSES = 12
EXPECTED_COLS = 6
FLOAT_TOL = 1e-9
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VERSION = "2.0 (2026-08-05) - with --images correspondence check"

log = logging.getLogger("verify_labels")


# --------------------------------------------------------------------------- #
# Reading / normalization
# --------------------------------------------------------------------------- #
def read_normalized(path: Path) -> tuple[list[str], bool]:
    """Read a label file, return (lines without EOL junk, had_crlf_flag).

    Normalization: CRLF/CR -> LF, strip trailing whitespace per line,
    drop trailing empty lines. Original file is NOT modified.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    had_crlf = "\r" in raw
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines, had_crlf


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@dataclass
class FileIssues:
    """All problems found in a single label file."""

    path: Path
    bad_col_lines: list[tuple[int, int]] = field(default_factory=list)
    parse_errors: list[tuple[int, str]] = field(default_factory=list)
    range_errors: list[tuple[int, str]] = field(default_factory=list)
    extent_warnings: list[tuple[int, str]] = field(default_factory=list)
    had_crlf: bool = False
    is_empty: bool = False

    @property
    def has_errors(self) -> bool:
        return bool(self.bad_col_lines or self.parse_errors or self.range_errors)


@dataclass
class Stats:
    """Aggregate dataset statistics."""

    files: int = 0
    empty_files: int = 0
    crlf_files: int = 0
    lines: int = 0
    class_counts: dict[int, int] = field(default_factory=dict)
    damage_counts: dict[int, int] = field(default_factory=dict)


def validate_file(path: Path, stats: Stats,
                  num_classes: int = DEFAULT_NUM_CLASSES) -> FileIssues:
    """Validate one label file against the 6-column YOLO+flag contract."""
    issues = FileIssues(path=path)
    lines, issues.had_crlf = read_normalized(path)
    stats.files += 1
    if issues.had_crlf:
        stats.crlf_files += 1
    if not lines:
        issues.is_empty = True
        stats.empty_files += 1
        return issues

    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        tokens = line.split()
        stats.lines += 1
        if len(tokens) != EXPECTED_COLS:
            issues.bad_col_lines.append((i, len(tokens)))
            continue
        try:
            cls = int(tokens[0])
            coords = [float(t) for t in tokens[1:5]]
            flag = int(tokens[5])
        except ValueError as exc:
            issues.parse_errors.append((i, str(exc)))
            continue

        if not 0 <= cls < num_classes:
            issues.range_errors.append((i, f"class_id {cls} outside [0,{num_classes - 1}]"))
        for name, v in zip(("cx", "cy", "w", "h"), coords):
            if not 0.0 <= v <= 1.0:
                issues.range_errors.append((i, f"{name}={v} outside [0,1]"))
        if flag not in (0, 1):
            issues.range_errors.append((i, f"damage_flag {flag} not in {{0,1}}"))

        cx, cy, w, h = coords
        for edge, val in (("left", cx - w / 2), ("right", cx + w / 2),
                          ("top", cy - h / 2), ("bottom", cy + h / 2)):
            if val < -1e-6 or val > 1 + 1e-6:
                issues.extent_warnings.append((i, f"box {edge} edge at {val:.4f}"))

        stats.class_counts[cls] = stats.class_counts.get(cls, 0) + 1
        if flag in (0, 1):
            stats.damage_counts[flag] = stats.damage_counts.get(flag, 0) + 1
    return issues


# --------------------------------------------------------------------------- #
# Comparison local vs branch
# --------------------------------------------------------------------------- #
def parse_numeric(lines: list[str]) -> list[list[float]] | None:
    """Parse lines to float token lists; None if any token is non-numeric."""
    out: list[list[float]] = []
    for ln in lines:
        if not ln.strip():
            continue
        try:
            out.append([float(t) for t in ln.split()])
        except ValueError:
            return None
    return out


def numerically_equal(a: list[str], b: list[str]) -> bool:
    """True when two files carry identical numbers despite formatting diffs."""
    pa, pb = parse_numeric(a), parse_numeric(b)
    if pa is None or pb is None or len(pa) != len(pb):
        return False
    for ra, rb in zip(pa, pb):
        if len(ra) != len(rb):
            return False
        if any(abs(x - y) > FLOAT_TOL for x, y in zip(ra, rb)):
            return False
    return True


@dataclass
class CompareResult:
    missing_locally: list[str] = field(default_factory=list)
    differing_content: list[str] = field(default_factory=list)
    formatting_only: list[str] = field(default_factory=list)
    local_only: list[str] = field(default_factory=list)
    identical: int = 0


def compare_dirs(local: Path, branch: Path) -> CompareResult:
    """Compare *.txt files between local labels dir and branch-extracted dir."""
    res = CompareResult()
    local_files = {p.name: p for p in sorted(local.glob("*.txt"))}
    branch_files = {p.name: p for p in sorted(branch.glob("*.txt"))}

    for name, bpath in branch_files.items():
        lpath = local_files.get(name)
        if lpath is None:
            res.missing_locally.append(name)
            continue
        llines, _ = read_normalized(lpath)
        blines, _ = read_normalized(bpath)
        if llines == blines:
            res.identical += 1
        elif numerically_equal(llines, blines):
            res.formatting_only.append(name)
        else:
            res.differing_content.append(name)

    res.local_only = sorted(set(local_files) - set(branch_files))
    return res


# --------------------------------------------------------------------------- #
# Image <-> label correspondence
# --------------------------------------------------------------------------- #
@dataclass
class Correspondence:
    """Orphans in both directions between images dir and labels dir."""

    images_without_labels: list[str] = field(default_factory=list)
    labels_without_images: list[str] = field(default_factory=list)
    matched: int = 0


def check_correspondence(images_dir: Path, labels_dir: Path) -> Correspondence:
    """Match image stems against label stems and report orphans both ways."""
    img_stems = {p.stem for p in images_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in IMG_EXTS}
    lbl_stems = {p.stem for p in labels_dir.glob("*.txt")}
    return Correspondence(
        images_without_labels=sorted(img_stems - lbl_stems),
        labels_without_images=sorted(lbl_stems - img_stems),
        matched=len(img_stems & lbl_stems),
    )


# --------------------------------------------------------------------------- #
# EOL fixing
# --------------------------------------------------------------------------- #
def fix_eol(labels_dir: Path) -> int:
    """Rewrite every *.txt with LF endings and stripped trailing whitespace."""
    changed = 0
    for path in sorted(labels_dir.glob("*.txt")):
        lines, had_crlf = read_normalized(path)
        new_text = "\n".join(lines) + ("\n" if lines else "")
        if had_crlf or new_text != path.read_text(encoding="utf-8", errors="replace"):
            path.write_text(new_text, encoding="utf-8", newline="\n")
            changed += 1
    return changed


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _render_list(items: list[str], cap: int | None) -> list[str]:
    """Render item lines; truncate at cap for console output (None = full)."""
    shown = items if cap is None else items[:cap]
    lines = [f"  - `{x}`" for x in shown]
    if cap is not None and len(items) > cap:
        lines.append(f"  - ... and {len(items) - cap} more (full list in the report file)")
    return lines


def build_report(all_issues: list[FileIssues], stats: Stats,
                 cmp_res: CompareResult | None,
                 corr: Correspondence | None = None,
                 num_classes: int = DEFAULT_NUM_CLASSES,
                 labels_dir: Path | None = None,
                 cap: int | None = None) -> str:
    """Render a markdown gate report."""
    bad = [i for i in all_issues if i.has_errors]
    lines: list[str] = [
        "# Phase 0 gate report",
        "",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- labels dir: `{labels_dir}`" if labels_dir else "- labels dir: (not recorded)",
        f"- num_classes: {num_classes}",
        "",
    ]

    lines += [
        "## Dataset statistics",
        f"- label files scanned: **{stats.files}** "
        f"(empty: {stats.empty_files}, CRLF endings: {stats.crlf_files})",
        f"- object lines: **{stats.lines}**",
        f"- damage flag counts: {dict(sorted(stats.damage_counts.items()))}",
        f"- per-class boxes: {dict(sorted(stats.class_counts.items()))}",
        "",
        "## Validation",
    ]
    if not bad:
        lines.append("All files conform to the 6-column contract. PASS")
    else:
        lines.append(f"**{len(bad)} file(s) with errors:**")
        for iss in bad:
            lines.append(f"- `{iss.path.name}`")
            for ln, n in iss.bad_col_lines:
                lines.append(f"  - line {ln}: {n} columns (expected {EXPECTED_COLS})")
            for ln, msg in iss.parse_errors:
                lines.append(f"  - line {ln}: parse error: {msg}")
            for ln, msg in iss.range_errors:
                lines.append(f"  - line {ln}: {msg}")
    warn = [i for i in all_issues if i.extent_warnings]
    if warn:
        lines += ["", f"Box-extent warnings (clamped later by the cropper) in {len(warn)} file(s) - informational only."]

    if cmp_res is not None:
        lines += ["", "## Local vs branch comparison"]
        lines.append(f"- identical: {cmp_res.identical}")
        lines.append(f"- formatting-only differences (safe): {len(cmp_res.formatting_only)}")
        lines.append(f"- local-only files (newer work, fine): {len(cmp_res.local_only)}")
        if cmp_res.differing_content:
            lines.append(f"- **content differs ({len(cmp_res.differing_content)}) - review each, keep the correct one:**")
            lines += _render_list(cmp_res.differing_content, cap)
        if cmp_res.missing_locally:
            lines.append(f"- **MISSING LOCALLY ({len(cmp_res.missing_locally)}) - copy these from the branch before deleting it:**")
            lines += _render_list(cmp_res.missing_locally, cap)
        else:
            lines.append("- missing locally: none. Branch is safe to delete once validation passes.")

    if corr is not None:
        lines += ["", "## Image <-> label correspondence"]
        lines.append(f"- matched pairs: {corr.matched}")
        if corr.labels_without_images:
            lines.append(f"- **orphan labels ({len(corr.labels_without_images)}) - "
                         "label exists but image is missing (ERROR, investigate):**")
            lines += _render_list(corr.labels_without_images, cap)
        if corr.images_without_labels:
            lines.append(f"- **images without a label file ({len(corr.images_without_labels)}) - "
                         "if an image truly has no objects, create an EMPTY .txt for it "
                         "(YOLO negative sample); otherwise annotate or remove it "
                         "BEFORE tagging data-v1.0:**")
            lines += _render_list(corr.images_without_labels, cap)
        if not corr.labels_without_images and not corr.images_without_labels:
            lines.append("- 1:1 correspondence holds in both directions. OK")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True, type=Path,
                        help="Your LOCAL authoritative labels dir (data/annotations/labels)")
    parser.add_argument("--compare", type=Path, default=None,
                        help="Branch-extracted labels dir to compare against")
    parser.add_argument("--images", type=Path, default=None,
                        help="Raw images dir (data/raw) to check image<->label correspondence")
    parser.add_argument("--report", type=Path, default=Path("verification_report.md"),
                        help="Markdown report path (always written; default: verification_report.md)")
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES,
                        help="valid class_id range is [0, num_classes-1]")
    parser.add_argument("--fix-eol", action="store_true",
                        help="Rewrite local files with LF endings (in-place)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log.info("verify_labels version %s", VERSION)

    if not args.labels.is_dir():
        log.error("labels dir not found: %s", args.labels)
        return 1

    if args.fix_eol:
        n = fix_eol(args.labels)
        log.info("normalized line endings in %d file(s)", n)

    stats = Stats()
    all_issues = [validate_file(p, stats, args.num_classes)
                  for p in sorted(args.labels.glob("*.txt"))]
    bad = [i for i in all_issues if i.has_errors]

    cmp_res: CompareResult | None = None
    if args.compare is not None:
        if not args.compare.is_dir():
            log.error("compare dir not found: %s", args.compare)
            return 1
        cmp_res = compare_dirs(args.labels, args.compare)

    corr: Correspondence | None = None
    if args.images is not None:
        if not args.images.is_dir():
            log.error("images dir not found: %s", args.images)
            return 1
        corr = check_correspondence(args.images, args.labels)

    common = dict(num_classes=args.num_classes, labels_dir=args.labels)
    print("\n" + build_report(all_issues, stats, cmp_res, corr, cap=30, **common))
    args.report.write_text(
        build_report(all_issues, stats, cmp_res, corr, cap=None, **common),
        encoding="utf-8")
    log.info("full report written to %s", args.report)

    code = 0
    if bad:
        code |= 1
    if cmp_res and cmp_res.missing_locally:
        code |= 2
    if corr and corr.labels_without_images:
        code |= 4
    log.info("GATE %s (exit code %d)", "PASS" if code == 0 else "FAIL", code)
    return code


if __name__ == "__main__":
    sys.exit(main())
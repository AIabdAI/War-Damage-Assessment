# Annotation quality evidence — for Results §x.1 (dataset credibility)

Answers the standard examiner question *"how do you know your labels are
correct?"* with quantified, file-backed evidence:

| QA mechanism | evidence | file |
|---|---|---|
| Automated validation gate | Before cleanup (2026-08-08, 12-class era): 8,852 files scanned, **74 files with errors** (out-of-range widths, illegal class ids). After cleanup + 11-class migration: 9,964 files, **0 errors — PASS** | `validation_gate_FAIL_before_cleanup.md` → `validation_gate_PASS.md` |
| Human class review | **1,490 / 16,180 boxes manually verified**; 148 boxes in 133 files flagged wrong; **706 class corrections applied**; outcome: 20→11 class consolidation (Beam removed) | `class_review_report.md` |
| AI-assisted review | YOLO-World (yolov8l-worldv2, conf 0.35, IoU 0.45) auto-flagged **446 suspect boxes across 442 files** for human triage — a semi-automated annotation-QA loop | `reports/auto_class_suspects.json` (repo) |
| Annotation process | 4 annotators, lock/release coordination protocol, sprint 2026-07-05 → 07-12; per-annotator statistics | `annotation_report.md`, `annotation_stats.yaml` |
| Pipeline integrity | Zero data loss at every processing step: `split_skipped.json` and `crop_skipped.json` — all skip counters 0 | repo `reports/` |
| Contract tests | Label-format unit tests (`tests/`) + CI data-guard blocking raw data in git | repo |

Provenance census of the 9,964 raw images (state in the dataset section and
qualify the "single-region" limitation accordingly):

| source | images | share |
|---|---|---|
| Team-captured photos (MB_*) | 6,521 | 65.4 % |
| Own photos (other) | 2,814 | 28.2 % |
| Web/Roboflow-sourced | 472 | 4.7 % |
| Own video frames (VID_*) | 157 | 1.6 % |

Limitation to state honestly: inter-annotator agreement was never measured
(single-pass annotation with review, not double-annotation) — list under
limitations/future work.

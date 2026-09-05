# Phase 0 gate report

- generated: 2026-08-11T14:32:42
- labels dir: `data\annotations\labels`
- num_classes: 12

## Dataset statistics
- label files scanned: **11417** (empty: 259, CRLF endings: 0)
- object lines: **17207**
- damage flag counts: {0: 8105, 1: 9102}
- per-class boxes: {0: 4138, 1: 3666, 2: 899, 3: 872, 4: 762, 5: 636, 6: 669, 7: 1295, 8: 1604, 9: 89, 10: 2068, 11: 509}

## Validation
**8 file(s) with errors:**
- `MB_light_fixture_1236.txt`
  - line 1: h=1.020548 outside [0,1]
- `MB_Pillar_2324.txt`
  - line 3: w=1.006849 outside [0,1]
- `MB_Pillar_2344.txt`
  - line 1: w=1.009589 outside [0,1]
- `MB_Pillar_2350.txt`
  - line 1: w=1.013699 outside [0,1]
- `MB_Pillar_2429.txt`
  - line 1: h=1.00137 outside [0,1]
- `MB_Pillar_2430.txt`
  - line 1: h=1.00137 outside [0,1]
- `MB_Pillar_2431.txt`
  - line 1: h=1.00137 outside [0,1]
- `MB_wall_cabinet_0255.txt`
  - line 1: w=1.021918 outside [0,1]

Box-extent warnings (clamped later by the cropper) in 930 file(s) - informational only.

## Image <-> label correspondence
- matched pairs: 11417
- **images without a label file (7) - if an image truly has no objects, create an EMPTY .txt for it (YOLO negative sample); otherwise annotate or remove it BEFORE tagging data-v1.0:**
  - `bricks_597(1)`
  - `bricks_598(1)`
  - `bricks_599(1)`
  - `bricks_6(1)`
  - `bricks_60(1)`
  - `bricks_600(1)`
  - `bricks_601(1)`

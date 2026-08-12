#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
War-Damage-Assessment — Class Verification Tool
================================================
أداة مراجعة صحة الكلاسات: بعض الصناديق معلّمة بكلاس خطأ، وهذه الأداة تجعل
اكتشافها سهلاً بعرض قصاصات (crops) كل الصناديق مجمّعة حسب الكلاس — أي صندوق
"شاذ" وسط شبكة قصاصات من نفس الكلاس يُكتشف بنظرة واحدة.

- شبكة قصاصات لكل كلاس (مع عدّاد المراجَع/الكلي لكل كلاس).
- كلاس زائف «؟ خارج النطاق» يجمع أي صندوق cls خارج [0, 11] تلقائياً.
- نقرة على قصاصة تفتح نافذة المراجعة: الصورة كاملة مع الصندوق مظللاً.
- في نافذة المراجعة: Space = الكلاس صحيح وانتقل للتالي،
  E = وسم «⚠ كلاس خاطئ» (يُدرج الملف في التقرير للمراجعة لاحقاً في review_tool)،
  مفاتيح 1..9,0,q,w = تصحيح الكلاس (يُحفظ فوراً في labels/ و labels_obb/ معاً
  بتغيير رقم الكلاس فقط — الإحداثيات لا تُمَس)، Del = حذف الصندوق نهائياً.
- الحالة تُحفظ في class_review_state.json (استئناف المراجعة بين الجلسات).
- تصدير تقرير Markdown: reports/class_review_report.md — يشمل أسماء الملفات
  الموسومة «كلاس خاطئ» والكلاسات خارج النطاق بصيغة يقرؤها review_tool.py
  (الذي يقرأ verification_report.md وهذا التقرير معاً) لمراجعتها وإصلاحها.

الاستخدام (من جذر الريبو داخل البيئة الافتراضية):
    python class_check_tool.py                 # الواجهة الرسومية
    python class_check_tool.py --scan          # فحص سريع في الطرفية بلا واجهة
    python class_check_tool.py --auto          # فحص تلقائي بـ YOLO-World —
                                               #   إشارة فقط: يذكر الصور المشتبه
                                               #   بها في التقرير، ولا يعدّل أي
                                               #   حالة أو labels تلقائياً
    python class_check_tool.py --user Abdallah # تحديد اسم المراجِع

بعد الوسم (يدوياً أو بـ --auto) صدّر التقرير ثم راجع الملفات المدرجة فيه عبر:
    python review_tool.py      # يقرأ verification_report.md وتقرير الكلاسات معاً

المتطلبات: Pillow (موجودة في requirements.txt)؛ ولوضع --auto: ultralytics.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------------
# إعدادات متطابقة مع annotation_tool.py — نفس الكلاسات والألوان والمسارات
# ----------------------------------------------------------------------------

CLASSES = [
    "Brick_Wall",       # 0
    "Column",           # 1
    "Staircase",        # 2
    "Floor_Tiles",      # 3
    "Sink",             # 4
    "Wall_Cabinet",     # 5
    "Window",           # 6
    "Door",             # 7
    "Air_Conditioner",  # 8
    "Light_Fixture",    # 9
    "Toilet",           # 10
]

CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#f032e6", "#bcf60c", "#fabebe", "#008080",
    "#e6beff", "#9a6324",
]

CLASS_KEYS = list("1234567890qwertyuiop")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

RAW_DIR = Path("data/raw")
LABELS_DIR = Path("data/annotations/labels")
LABELS_OBB_DIR = Path("data/annotations/labels_obb")
STATE_FILE = Path("class_review_state.json")
REPORT_PATH = Path("reports/class_review_report.md")
AUTO_SUSPECTS_PATH = Path("reports/auto_class_suspects.json")

OUT_OF_RANGE = -1          # معرّف الكلاس الزائف «خارج النطاق» في الواجهة
THUMB = 150                # حجم القصاصة في الشبكة (بكسل)
GRID_COLS, GRID_ROWS = 8, 5
PAGE_SIZE = GRID_COLS * GRID_ROWS
CROP_PAD = 0.15            # هامش حول الصندوق عند القصّ (نسبة من حجمه)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------------
# قراءة ملفات الـ labels
#   box = dict(cls, dmg, rect=(x0,y0,x1,y1) منسّق 0..1, pts=[4 نقاط] أو None)
#   نقرأ labels_obb إن وُجد (المصدر الأكمل) وإلا labels القياسي.
# ----------------------------------------------------------------------------

def _line_box(parts: list[str]) -> dict | None:
    try:
        cls = int(float(parts[0]))
        if len(parts) in (9, 10):
            dmg = int(float(parts[9])) if len(parts) == 10 else 0
            vals = list(map(float, parts[1:9]))
            pts = [(vals[i], vals[i + 1]) for i in range(0, 8, 2)]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            return {"cls": cls, "dmg": dmg, "pts": pts,
                    "rect": (min(xs), min(ys), max(xs), max(ys))}
        if len(parts) in (5, 6):
            dmg = int(float(parts[5])) if len(parts) == 6 else 0
            cx, cy, w, h = map(float, parts[1:5])
            return {"cls": cls, "dmg": dmg, "pts": None,
                    "rect": (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)}
    except ValueError:
        return None
    return None


def _valid_line_indices(lines: list[str]) -> list[int]:
    """فهارس الأسطر الصالحة كصناديق — نفس معيار القراءة، كي يتطابق ترتيب
    الصندوق i مع السطر valid[i] عند الكتابة."""
    out = []
    for i, ln in enumerate(lines):
        parts = ln.split()
        if parts and _line_box(parts) is not None:
            out.append(i)
    return out


def label_file_for(repo: Path, stem: str) -> Path | None:
    for folder in (LABELS_OBB_DIR, LABELS_DIR):
        p = repo / folder / f"{stem}.txt"
        if p.exists():
            return p
    return None


def parse_label_file(path: Path) -> list[dict]:
    boxes = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return boxes
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        b = _line_box(parts)
        if b is not None:
            boxes.append(b)
    return boxes


def scan_boxes(repo: Path) -> tuple[list[dict], dict[str, Path]]:
    """يرجع (قائمة كل الصناديق، فهرس الصور).
    كل عنصر: dict(stem, idx, cls, dmg, rect, pts) — idx ترتيب الصندوق في ملفه."""
    raw = repo / RAW_DIR
    image_index = {p.stem: p for p in raw.iterdir()
                   if p.suffix.lower() in IMG_EXTS} if raw.exists() else {}
    refs: list[dict] = []
    seen: set[str] = set()
    for folder in (LABELS_OBB_DIR, LABELS_DIR):
        d = repo / folder
        if not d.exists():
            continue
        for path in sorted(d.glob("*.txt")):
            if path.stem in seen:
                continue
            seen.add(path.stem)
            for i, b in enumerate(parse_label_file(path)):
                refs.append({"stem": path.stem, "idx": i, **b})
    return refs, image_index


# ----------------------------------------------------------------------------
# تعديل الكلاس / حذف صندوق — في الملفين معاً، بتغيير الحد الأدنى من النص
# ----------------------------------------------------------------------------

def _edit_files(repo: Path, stem: str, box_idx: int, edit) -> list[str]:
    """يطبّق `edit(lines, line_index)` على كل ملف label موجود لهذا الـ stem.
    يرجع قائمة تحذيرات (ملف لا يحوي الصندوق المطلوب مثلاً)."""
    warnings = []
    for folder in (LABELS_OBB_DIR, LABELS_DIR):
        path = repo / folder / f"{stem}.txt"
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        valid = _valid_line_indices(lines)
        if box_idx >= len(valid):
            warnings.append(f"{folder.name}/{stem}.txt: لا يحوي صندوقاً رقم {box_idx}")
            continue
        edit(lines, valid[box_idx])
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return warnings


def change_class(repo: Path, stem: str, box_idx: int, new_cls: int) -> list[str]:
    def edit(lines, li):
        parts = lines[li].split()
        parts[0] = str(new_cls)
        lines[li] = " ".join(parts)
    return _edit_files(repo, stem, box_idx, edit)


def delete_box(repo: Path, stem: str, box_idx: int) -> list[str]:
    def edit(lines, li):
        del lines[li]
    return _edit_files(repo, stem, box_idx, edit)


# ----------------------------------------------------------------------------
# حالة المراجعة — class_review_state.json
# ----------------------------------------------------------------------------

def box_key(stem: str, idx: int) -> str:
    return f"{stem}#{idx}"


def _empty_state() -> dict:
    return {"verified": {}, "flagged": {}, "changes": [], "deleted": []}


def read_state(path: Path) -> dict:
    if not path.exists():
        return _empty_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_state()
    for key, default in _empty_state().items():
        data.setdefault(key, default)
    return data


def write_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def shift_state_after_delete(state: dict, stem: str, deleted_idx: int) -> None:
    """بعد حذف صندوق تنزاح فهارس ما بعده في نفس الملف — نحدّث مفاتيح
    verified و flagged معاً."""
    for section in ("verified", "flagged"):
        updated = {}
        for key, val in state[section].items():
            s, _, i = key.rpartition("#")
            if s == stem:
                i = int(i)
                if i == deleted_idx:
                    continue          # الصندوق المحذوف يسقط من السجل
                if i > deleted_idx:
                    key = box_key(s, i - 1)
            updated[key] = val
        state[section] = updated


# ----------------------------------------------------------------------------
# تقرير Markdown
# ----------------------------------------------------------------------------

def cls_name(c: int) -> str:
    return CLASSES[c] if 0 <= c < len(CLASSES) else f"؟({c})"


def export_report(repo: Path, state: dict, refs: list[dict]) -> Path:
    """يكتب reports/class_review_report.md.

    قسما «wrong class» و«out-of-range class» مكتوبان بصيغة يفهمها
    parse_report في review_tool.py (سطر يحوي العلامة، ثم `- \\`stem\\``،
    ثم أسباب تبدأ بـ `- line ...`) — فيعرض review_tool هذه الملفات للمراجعة.
    """
    per_class = {name: 0 for name in CLASSES}
    verified_per_class = {name: 0 for name in CLASSES}
    by_key = {box_key(r["stem"], r["idx"]): r for r in refs}
    for r in refs:
        if 0 <= r["cls"] < len(CLASSES):
            per_class[CLASSES[r["cls"]]] += 1
            if box_key(r["stem"], r["idx"]) in state["verified"]:
                verified_per_class[CLASSES[r["cls"]]] += 1

    # تجميع الموسومة «كلاس خاطئ» حسب الملف
    flagged_files: dict[str, list[str]] = {}
    for key, fl in sorted(state["flagged"].items()):
        stem, _, idx = key.rpartition("#")
        r = by_key.get(key)
        cname = cls_name(r["cls"]) if r else cls_name(fl.get("cls", -1))
        reason = f"line {int(idx) + 1}: class '{cname}' flagged wrong by {fl.get('by', '?')}"
        if "suggest" in fl:
            reason += (f" — suggests '{cls_name(fl['suggest'])}'"
                       f" (conf {fl.get('conf', 0):.2f})")
        flagged_files.setdefault(stem, []).append(reason)

    # الكلاسات خارج النطاق تُدرج تلقائياً
    oor_files: dict[str, list[str]] = {}
    for r in refs:
        if not (0 <= r["cls"] < len(CLASSES)):
            oor_files.setdefault(r["stem"], []).append(
                f"line {r['idx'] + 1}: class id {r['cls']} outside [0..{len(CLASSES) - 1}]")

    # اشتباهات الفحص التلقائي (YOLO-World) — إشارة فقط، من ملف الاشتباهات.
    # نستبعد ما راجعه إنسان لاحقاً (verified) وما تغيّر كلاسه منذ الفحص.
    auto_files: dict[str, list[str]] = {}
    auto_meta = ""
    auto_path = repo / AUTO_SUSPECTS_PATH
    if auto_path.exists():
        try:
            auto = json.loads(auto_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            auto = {}
        auto_meta = (f" (model: {auto.get('model', '?')}, "
                     f"generated: {auto.get('generated_at', '?')})")
        for stem, items in sorted(auto.get("suspects", {}).items()):
            for it in items:
                key = box_key(stem, it["idx"])
                if key in state["verified"]:
                    continue
                r = by_key.get(key)
                if r is None or r["cls"] != it["cls"]:
                    continue        # الصندوق حُذف أو صُحّح كلاسه بعد الفحص
                auto_files.setdefault(stem, []).append(
                    f"line {it['idx'] + 1}: class '{cls_name(it['cls'])}' — "
                    f"model sees '{cls_name(it['suggest'])}' (conf {it['conf']:.2f})")

    lines = [
        "# Class Review Report — War-Damage-Assessment",
        "",
        f"- **Generated:** {utcnow_iso()}",
        f"- **Boxes verified:** {len(state['verified'])} / {len(refs)}",
        f"- **Boxes flagged wrong:** {len(state['flagged'])} "
        f"(in {len(flagged_files)} files)",
        f"- **Class corrections:** {len(state['changes'])}",
        f"- **Boxes deleted:** {len(state['deleted'])}",
        "",
        "## ملفات فيها أنوتيشن كلاس خاطئ — راجعها عبر review_tool.py",
        "",
        f"{len(flagged_files)} file(s) with wrong class annotations:",
        "",
    ]
    for stem, reasons in sorted(flagged_files.items()):
        lines.append(f"- `{stem}`")
        lines += [f"  - {reason}" for reason in reasons]
    if not flagged_files:
        lines.append("_No flagged boxes._")
    lines += [
        "",
        f"## اشتباه تلقائي — YOLO-World{auto_meta}",
        "",
        "إشارة فقط: لم يُعدَّل أي label أو حالة تلقائياً — راجعها بنفسك.",
        "",
        f"{len(auto_files)} file(s) with wrong class (auto-detected):",
        "",
    ]
    for stem, reasons in sorted(auto_files.items()):
        lines.append(f"- `{stem}`")
        lines += [f"  - {reason}" for reason in reasons]
    if not auto_files:
        lines.append("_No auto-detected suspects._")
    lines += [
        "",
        "## كلاسات خارج النطاق",
        "",
        f"{len(oor_files)} file(s) with out-of-range class ids:",
        "",
    ]
    for stem, reasons in sorted(oor_files.items()):
        lines.append(f"- `{stem}`")
        lines += [f"  - {reason}" for reason in reasons]
    if not oor_files:
        lines.append("_No out-of-range class ids._")
    lines += [
        "",
        "## Progress per class",
        "",
        "| Class | Verified | Total |",
        "|---|---|---|",
    ]
    for name in CLASSES:
        lines.append(f"| {name} | {verified_per_class[name]} | {per_class[name]} |")
    lines += ["", "## Corrections", ""]
    if state["changes"]:
        lines += ["| Image | Box | From | To | By | At |", "|---|---|---|---|---|---|"]
        for ch in state["changes"]:
            lines.append(f"| `{ch['stem']}` | {ch['idx']} | {cls_name(ch['from'])} "
                         f"| {cls_name(ch['to'])} | {ch['by']} | {ch['at']} |")
    else:
        lines.append("_No corrections yet._")
    lines += ["", "## Deleted boxes", ""]
    if state["deleted"]:
        lines += ["| Image | Box | Class | By | At |", "|---|---|---|---|---|"]
        for d in state["deleted"]:
            lines.append(f"| `{d['stem']}` | {d['idx']} | {cls_name(d['cls'])} "
                         f"| {d['by']} | {d['at']} |")
    else:
        lines.append("_No deletions._")
    lines.append("")
    out = repo / REPORT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------
# فحص سريع بلا واجهة: توزيع الكلاسات + كلاسات خارج النطاق + صناديق بلا صورة
# ----------------------------------------------------------------------------

def cli_scan(repo: Path) -> int:
    refs, image_index = scan_boxes(repo)
    per_class: dict[int, int] = {}
    for r in refs:
        per_class[r["cls"]] = per_class.get(r["cls"], 0) + 1
    print(f"إجمالي الصناديق: {len(refs)} في {len({r['stem'] for r in refs})} صورة\n")
    print("التوزيع حسب الكلاس:")
    for c in sorted(per_class):
        print(f"  {c:>3} {cls_name(c):<18} {per_class[c]}")
    bad = [r for r in refs if not (0 <= r["cls"] < len(CLASSES))]
    if bad:
        print(f"\n⚠ صناديق بكلاس خارج النطاق [0..{len(CLASSES)-1}]: {len(bad)}")
        for r in bad[:50]:
            print(f"  - {r['stem']} (box {r['idx']}): cls={r['cls']}")
        if len(bad) > 50:
            print(f"  ... و{len(bad) - 50} أخرى")
    no_img = sorted({r["stem"] for r in refs} - set(image_index))
    if no_img:
        print(f"\n⚠ labels بلا صورة مقابلة في data/raw: {len(no_img)}")
        for s in no_img[:20]:
            print(f"  - {s}")
        if len(no_img) > 20:
            print(f"  ... و{len(no_img) - 20} أخرى")
    if not bad and not no_img:
        print("\n✅ لا كلاسات خارج النطاق ولا labels يتيمة — راجع الكلاسات بصرياً عبر الواجهة.")
    return 0


# ----------------------------------------------------------------------------
# فحص تلقائي بـ YOLO-World (open-vocabulary) — إشارة فقط، بلا أي إجراء تلقائي:
#   النموذج يتنبأ بالكلاسات من أوصاف نصية، ونقارن تنبؤه بكلاس كل صندوق GT.
#   صندوق GT لا يطابقه (IoU) أي تنبؤ من كلاسه، بينما يطابقه تنبؤ واثق من كلاس
#   آخر → مشتبه به: يُكتب في reports/auto_class_suspects.json وفي التقرير
#   فقط. لا يُعدَّل أي label ولا حالة المراجعة — القرار كله للمراجِع البشري.
# ----------------------------------------------------------------------------

# أوصاف نصية مفهومة للنموذج — بترتيب CLASSES نفسه
YOLO_WORLD_PROMPTS = [
    "brick wall",        # Brick_Wall
    "concrete column",   # Column
    "staircase",         # Staircase
    "tiled floor",       # Floor_Tiles
    "sink",              # Sink
    "wall cabinet",      # Wall_Cabinet
    "window",            # Window
    "door",              # Door
    "air conditioner",   # Air_Conditioner
    "light fixture",     # Light_Fixture
    "toilet",            # Toilet
]


def _iou(a: tuple, b: tuple) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def auto_check(repo: Path, model_name: str, conf: float, iou_thr: float,
               device: str | None, limit: int | None) -> int:
    try:
        from ultralytics import YOLOWorld
    except ImportError:
        print("وضع --auto يتطلب ultralytics:  pip install ultralytics", file=sys.stderr)
        return 1

    refs, image_index = scan_boxes(repo)
    by_stem: dict[str, list[dict]] = {}
    for r in refs:
        if 0 <= r["cls"] < len(CLASSES):        # خارج النطاق يُدرج في التقرير أصلاً
            by_stem.setdefault(r["stem"], []).append(r)
    stems = sorted(s for s in by_stem if s in image_index)
    if limit:
        stems = stems[:limit]
    print(f"فحص تلقائي: {len(stems)} صورة، النموذج: {model_name}")

    model = YOLOWorld(model_name)
    model.set_classes(YOLO_WORLD_PROMPTS)

    suspects: dict[str, list[dict]] = {}
    n_suspects = 0
    BATCH = 16
    for i in range(0, len(stems), BATCH):
        batch = stems[i:i + BATCH]
        results = model.predict([str(image_index[s]) for s in batch],
                                conf=min(conf, 0.05), verbose=False,
                                device=device)
        for stem, res in zip(batch, results):
            preds = []
            for box in res.boxes:
                preds.append((int(box.cls.item()), float(box.conf.item()),
                              tuple(box.xyxyn[0].tolist())))
            for r in by_stem[stem]:
                gt_rect = r["rect"]
                # أي تنبؤ من نفس الكلاس يغطي الصندوق (ولو بثقة منخفضة) → سليم
                same = max((iou for c, cf, xy in preds
                            if c == r["cls"] and (iou := _iou(gt_rect, xy)) > 0),
                           default=0.0)
                if same >= iou_thr:
                    continue
                # أفضل تنبؤ واثق من كلاس مختلف يطابق الصندوق → مشتبه به
                best = max(((cf, c) for c, cf, xy in preds
                            if c != r["cls"] and cf >= conf
                            and _iou(gt_rect, xy) >= iou_thr),
                           default=None)
                if best is None:
                    continue
                cf, c = best
                suspects.setdefault(stem, []).append(
                    {"idx": r["idx"], "cls": r["cls"],
                     "suggest": c, "conf": round(cf, 3)})
                n_suspects += 1
                print(f"  ⚠ {stem} (box {r['idx']}): "
                      f"{cls_name(r['cls'])} — النموذج يرى '{cls_name(c)}' ({cf:.2f})")
        print(f"  ... {min(i + BATCH, len(stems))}/{len(stems)} صورة")

    # إشارة فقط: تُكتب الاشتباهات في ملف جانبي + التقرير — لا حالة ولا labels
    out_json = repo / AUTO_SUSPECTS_PATH
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        {"generated_at": utcnow_iso(), "model": model_name,
         "conf": conf, "iou": iou_thr, "suspects": suspects},
        indent=2, ensure_ascii=False), encoding="utf-8")

    out = export_report(repo, read_state(repo / STATE_FILE), refs)
    print(f"\n⚠ المشتبه بها: {n_suspects} صندوق في {len(suspects)} صورة")
    for stem in sorted(suspects):
        print(f"  - {stem}")
    print(f"\n📄 قائمة الاشتباهات: {out_json}")
    print(f"📄 التقرير: {out}")
    print("لم يُعدَّل أي شيء تلقائياً — راجع الصور المذكورة عبر:  python review_tool.py")
    return 0


# ----------------------------------------------------------------------------
# الواجهة الرسومية
# ----------------------------------------------------------------------------

import tkinter as tk
from tkinter import messagebox

try:
    from PIL import Image, ImageTk, ImageDraw
except ImportError:
    print("مطلوب تثبيت Pillow:  pip install pillow", file=sys.stderr)
    raise


class ClassCheckApp:
    def __init__(self, root: tk.Tk, repo: Path, user: str):
        self.root = root
        self.repo = repo
        self.user = user
        self.state_path = repo / STATE_FILE
        self.state = read_state(self.state_path)

        self.root.title(f"WDA Class Check — {user}")
        self.root.geometry("1400x900")

        self.refs, self.image_index = scan_boxes(repo)
        if not self.refs:
            messagebox.showerror("لا بيانات",
                                 "لم أجد أي صناديق في data/annotations — "
                                 "تأكد أنك تشغّل الأداة من جذر الريبو.")
            root.destroy()
            return

        # اشتباهات الفحص التلقائي — للعرض فقط (🤖)، لا تُغيّر أي شيء
        self.auto_suspects: dict[str, dict] = {}
        auto_path = repo / AUTO_SUSPECTS_PATH
        if auto_path.exists():
            try:
                data = json.loads(auto_path.read_text(encoding="utf-8"))
                for stem, items in data.get("suspects", {}).items():
                    for it in items:
                        self.auto_suspects[box_key(stem, it["idx"])] = it
            except (json.JSONDecodeError, OSError):
                pass

        self.current_cls = 0
        self.page = 0
        self.only_unverified = tk.BooleanVar(value=False)
        self._thumbs: list = []          # مراجع PhotoImage (وإلا يجمعها GC)
        self._page_refs: list[dict] = []
        self.detail: DetailWindow | None = None

        self._build_ui()
        self.select_class(self._first_interesting_class())

    # ----------------------------------------------------------------- بيانات

    def _first_interesting_class(self) -> int:
        if any(not (0 <= r["cls"] < len(CLASSES)) for r in self.refs):
            return OUT_OF_RANGE
        return 0

    def class_refs(self, cls: int) -> list[dict]:
        if cls == OUT_OF_RANGE:
            return [r for r in self.refs if not (0 <= r["cls"] < len(CLASSES))]
        return [r for r in self.refs if r["cls"] == cls]

    def filtered_refs(self) -> list[dict]:
        refs = self.class_refs(self.current_cls)
        if self.only_unverified.get():
            refs = [r for r in refs
                    if box_key(r["stem"], r["idx"]) not in self.state["verified"]]
        return refs

    def is_verified(self, r: dict) -> bool:
        return box_key(r["stem"], r["idx"]) in self.state["verified"]

    def is_flagged(self, r: dict) -> bool:
        return box_key(r["stem"], r["idx"]) in self.state["flagged"]

    # ---------------------------------------------------------------------- UI

    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=4, pady=3)
        self.progress_var = tk.StringVar()
        tk.Label(top, textvariable=self.progress_var,
                 font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(top, text="غير المراجَع فقط", variable=self.only_unverified,
                       command=self.refresh_page).pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="📄 تصدير تقرير التصحيحات",
                  command=self.on_export_report).pack(side=tk.RIGHT, padx=4)
        tk.Button(top, text="✓ اعتماد كل صناديق الصفحة", bg="#d1f0d1",
                  command=self.verify_page).pack(side=tk.RIGHT, padx=4)

        # أزرار الكلاسات مع العدّادات
        cls_bar = tk.Frame(self.root)
        cls_bar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        self.class_btns: dict[int, tk.Button] = {}
        for i, name in enumerate(CLASSES):
            fg = "#000" if i in (7, 8, 10) else "#fff"
            b = tk.Button(cls_bar, text=name, bg=CLASS_COLORS[i], fg=fg,
                          font=("TkDefaultFont", 8),
                          command=lambda i=i: self.select_class(i))
            b.grid(row=i // 6, column=i % 6, sticky="ew", padx=1, pady=1)
            self.class_btns[i] = b
        b = tk.Button(cls_bar, text="؟ خارج النطاق", bg="#555", fg="#fff",
                      font=("TkDefaultFont", 8),
                      command=lambda: self.select_class(OUT_OF_RANGE))
        b.grid(row=0, column=6, rowspan=2, sticky="nsew", padx=1, pady=1)
        self.class_btns[OUT_OF_RANGE] = b
        for c in range(7):
            cls_bar.columnconfigure(c, weight=1)

        # شبكة القصاصات
        self.grid_frame = tk.Frame(self.root, bg="#2b2b2b")
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        nav = tk.Frame(self.root)
        nav.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=3)
        tk.Button(nav, text="⟨⟨ السابقة", command=lambda: self.set_page(self.page - 1)).pack(side=tk.LEFT)
        self.page_var = tk.StringVar()
        tk.Label(nav, textvariable=self.page_var).pack(side=tk.LEFT, padx=8)
        tk.Button(nav, text="التالية ⟩⟩", command=lambda: self.set_page(self.page + 1)).pack(side=tk.LEFT)
        self.status = tk.StringVar(value="انقر على أي قصاصة لمراجعتها — "
                                         "القصاصة الشاذة وسط كلاسها تعني كلاساً خاطئاً غالباً")
        tk.Label(nav, textvariable=self.status, anchor="e",
                 fg="#555").pack(side=tk.RIGHT, fill=tk.X, expand=True)

        self.root.bind("<Prior>", lambda e: self.set_page(self.page - 1))
        self.root.bind("<Next>", lambda e: self.set_page(self.page + 1))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _update_class_buttons(self):
        for cls, btn in self.class_btns.items():
            refs = self.class_refs(cls)
            done = sum(1 for r in refs if self.is_verified(r))
            name = "؟ خارج النطاق" if cls == OUT_OF_RANGE else CLASSES[cls]
            btn.configure(text=f"{name}  {done}/{len(refs)}",
                          relief=tk.SUNKEN if cls == self.current_cls else tk.RAISED)
        total_done = len(self.state["verified"])
        self.progress_var.set(f"المراجَع: {total_done}/{len(self.refs)} صندوق  |  "
                              f"⚠ موسوم خطأ: {len(self.state['flagged'])}  |  "
                              f"تصحيحات: {len(self.state['changes'])}  |  "
                              f"محذوف: {len(self.state['deleted'])}")

    # ------------------------------------------------------------------ الشبكة

    def select_class(self, cls: int):
        self.current_cls = cls
        self.page = 0
        self.refresh_page()

    def set_page(self, page: int):
        refs = self.filtered_refs()
        pages = max(1, (len(refs) + PAGE_SIZE - 1) // PAGE_SIZE)
        self.page = min(max(0, page), pages - 1)
        self.refresh_page(keep_page=True)

    def refresh_page(self, keep_page: bool = False):
        if not keep_page:
            self.page = min(self.page, max(0, (len(self.filtered_refs()) - 1) // PAGE_SIZE))
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self._thumbs.clear()

        refs = self.filtered_refs()
        pages = max(1, (len(refs) + PAGE_SIZE - 1) // PAGE_SIZE)
        self.page_var.set(f"صفحة {self.page + 1}/{pages}  ({len(refs)} صندوق)")
        self._page_refs = refs[self.page * PAGE_SIZE:(self.page + 1) * PAGE_SIZE]

        for k, r in enumerate(self._page_refs):
            img = self._crop_thumb(r)
            border = CLASS_COLORS[r["cls"]] if 0 <= r["cls"] < len(CLASSES) else "#ff0000"
            if self.is_flagged(r):
                hl = "#ff9800"
            elif self.is_verified(r):
                hl = "#00c800"
            else:
                hl = border
            cell = tk.Frame(self.grid_frame, bg=border, bd=0,
                            highlightthickness=2, highlightbackground=hl)
            cell.grid(row=k // GRID_COLS, column=k % GRID_COLS, padx=3, pady=3)
            if img is not None:
                tkimg = ImageTk.PhotoImage(img)
                self._thumbs.append(tkimg)
                lbl = tk.Label(cell, image=tkimg, bd=0)
            else:
                lbl = tk.Label(cell, text="صورة\nمفقودة", width=18, height=8,
                               bg="#444", fg="#fff")
            lbl.pack()
            if self.is_flagged(r):
                mark = "⚠ "
            elif self.is_verified(r):
                mark = "✔ "
            elif box_key(r["stem"], r["idx"]) in self.auto_suspects:
                mark = "🤖 "
            else:
                mark = ""
            cap = f"{mark}{cls_name(r['cls'])}"
            tk.Label(cell, text=cap, bg=border,
                     fg="#000" if r["cls"] in (7, 8, 10) else "#fff",
                     font=("TkDefaultFont", 7)).pack(fill=tk.X)
            for w in (lbl, cell):
                w.bind("<Button-1>", lambda e, k=k: self.open_detail(k))

        for c in range(GRID_COLS):
            self.grid_frame.columnconfigure(c, weight=1)
        self._update_class_buttons()

    def _crop_thumb(self, r: dict) -> Image.Image | None:
        path = self.image_index.get(r["stem"])
        if path is None:
            return None
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                W, H = im.size
                x0, y0, x1, y1 = r["rect"]
                pw, ph = (x1 - x0) * CROP_PAD, (y1 - y0) * CROP_PAD
                box = (max(0, int((x0 - pw) * W)), max(0, int((y0 - ph) * H)),
                       min(W, int((x1 + pw) * W)), min(H, int((y1 + ph) * H)))
                if box[2] <= box[0] or box[3] <= box[1]:
                    return None
                crop = im.crop(box)
                crop.thumbnail((THUMB, THUMB))
                canvas = Image.new("RGB", (THUMB, THUMB), "#1e1e1e")
                canvas.paste(crop, ((THUMB - crop.width) // 2, (THUMB - crop.height) // 2))
                return canvas
        except OSError:
            return None

    # ----------------------------------------------------------- أفعال المراجعة

    def mark_verified(self, r: dict):
        key = box_key(r["stem"], r["idx"])
        self.state["verified"][key] = {
            "cls": r["cls"], "by": self.user, "at": utcnow_iso()}
        self.state["flagged"].pop(key, None)   # المصحَّح/المؤكَّد لم يعد موسوماً
        write_state(self.state_path, self.state)

    def apply_flag(self, r: dict):
        """وسم الصندوق «كلاس خاطئ» — يُدرج ملفه في التقرير ليراجعه review_tool."""
        key = box_key(r["stem"], r["idx"])
        if key in self.state["flagged"]:
            self.state["flagged"].pop(key)     # ضغطة ثانية تلغي الوسم
            self.status.set(f"أُلغي وسم {r['stem']} (box {r['idx']})")
        else:
            self.state["flagged"][key] = {
                "cls": r["cls"], "by": self.user, "at": utcnow_iso()}
            self.state["verified"].pop(key, None)
            self.status.set(f"⚠ وُسم {r['stem']} (box {r['idx']}) ككلاس خاطئ — "
                            "سيظهر في التقرير")
        write_state(self.state_path, self.state)

    def verify_page(self):
        for r in self._page_refs:
            self.state["verified"][box_key(r["stem"], r["idx"])] = {
                "cls": r["cls"], "by": self.user, "at": utcnow_iso()}
        write_state(self.state_path, self.state)
        self.refresh_page(keep_page=True)
        self.status.set(f"✓ اعتُمدت {len(self._page_refs)} صناديق في هذه الصفحة")

    def apply_class_change(self, r: dict, new_cls: int) -> bool:
        if new_cls == r["cls"]:
            self.mark_verified(r)
            return True
        old = r["cls"]
        warnings = change_class(self.repo, r["stem"], r["idx"], new_cls)
        if warnings:
            messagebox.showwarning("تحذير", "\n".join(warnings))
        r["cls"] = new_cls
        self.state["changes"].append({"stem": r["stem"], "idx": r["idx"],
                                      "from": old, "to": new_cls,
                                      "by": self.user, "at": utcnow_iso()})
        self.mark_verified(r)
        self.status.set(f"✏ {r['stem']} (box {r['idx']}): "
                        f"{cls_name(old)} ← {cls_name(new_cls)}")
        return True

    def apply_delete(self, r: dict) -> bool:
        if not messagebox.askyesno(
                "حذف صندوق",
                f"حذف الصندوق {r['idx']} ({cls_name(r['cls'])}) نهائياً من labels "
                f"الصورة {r['stem']}؟"):
            return False
        warnings = delete_box(self.repo, r["stem"], r["idx"])
        if warnings:
            messagebox.showwarning("تحذير", "\n".join(warnings))
        self.state["deleted"].append({"stem": r["stem"], "idx": r["idx"],
                                      "cls": r["cls"], "by": self.user,
                                      "at": utcnow_iso()})
        shift_state_after_delete(self.state, r["stem"], r["idx"])
        write_state(self.state_path, self.state)
        # تحديث الفهارس في الذاكرة: إزالة الصندوق وإزاحة ما بعده في نفس الصورة
        self.refs.remove(r)
        for other in self.refs:
            if other["stem"] == r["stem"] and other["idx"] > r["idx"]:
                other["idx"] -= 1
        self.status.set(f"🗑 حُذف الصندوق {r['idx']} من {r['stem']}")
        return True

    # ------------------------------------------------------------ نافذة المراجعة

    def open_detail(self, k: int):
        refs = self.filtered_refs()
        pos = self.page * PAGE_SIZE + k
        if pos >= len(refs):
            return
        if self.detail is not None and self.detail.alive:
            self.detail.close()
        self.detail = DetailWindow(self, refs, pos)

    def on_export_report(self):
        out = export_report(self.repo, self.state, self.refs)
        self.status.set(f"📄 حُفظ التقرير: {out}")
        messagebox.showinfo("تقرير", f"حُفظ تقرير المراجعة:\n{out}")

    def on_close(self):
        write_state(self.state_path, self.state)
        self.root.destroy()


class DetailWindow:
    """نافذة مراجعة صندوق واحد: الصورة كاملة + الصندوق مظلل + تصحيح سريع."""

    CANVAS_W, CANVAS_H = 980, 720

    def __init__(self, app: ClassCheckApp, refs: list[dict], pos: int):
        self.app = app
        self.refs = refs
        self.pos = pos
        self.alive = True

        self.win = tk.Toplevel(app.root)
        self.win.geometry("1280x820")
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self.banner = tk.StringVar()
        tk.Label(self.win, textvariable=self.banner, font=("Segoe UI", 11, "bold"),
                 anchor="w").pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        body = tk.Frame(self.win)
        body.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, bg="#222",
                                width=self.CANVAS_W, height=self.CANVAS_H)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = tk.Frame(body, width=270)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)
        side.pack_propagate(False)

        tk.Label(side, text="هل الكلاس صحيح؟", font=("Segoe UI", 10, "bold")).pack(anchor="e")
        tk.Button(side, text="✓ صحيح — التالي (Space)", bg="#d1f0d1",
                  command=self.confirm).pack(fill=tk.X, pady=3)
        tk.Button(side, text="⚠ كلاس خاطئ — أدرجه في التقرير (E)", bg="#ffe0b3",
                  command=self.flag).pack(fill=tk.X, pady=3)
        tk.Label(side, text="أو اختر الكلاس الصحيح:").pack(anchor="e", pady=(8, 2))
        for i, name in enumerate(CLASSES):
            fg = "#000" if i in (7, 8, 10) else "#fff"
            tk.Button(side, text=f"{CLASS_KEYS[i]}  {name}", bg=CLASS_COLORS[i], fg=fg,
                      anchor="w", font=("TkDefaultFont", 8),
                      command=lambda i=i: self.reassign(i)).pack(fill=tk.X, pady=1)
        tk.Button(side, text="🗑 حذف الصندوق (Del)", bg="#f5b7b1",
                  command=self.delete).pack(fill=tk.X, pady=(10, 3))
        nav = tk.Frame(side)
        nav.pack(fill=tk.X, pady=6)
        tk.Button(nav, text="⟨ السابق (A)", command=self.prev).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(nav, text="(D) التالي ⟩", command=self.next).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Label(side, text="Space = صحيح وتقدّم • E = وسم خطأ\n"
                            "1..9,0,q,w = تصحيح الكلاس\n"
                            "A/D أو الأسهم = تنقّل • Esc = إغلاق",
                 justify="right", fg="#555", font=("TkDefaultFont", 8)).pack(side=tk.BOTTOM, pady=4)

        for i, key in enumerate(CLASS_KEYS[:len(CLASSES)]):
            self.win.bind(key, lambda e, i=i: self.reassign(i))
            self.win.bind(key.upper(), lambda e, i=i: self.reassign(i))
        self.win.bind("<space>", lambda e: self.confirm())
        self.win.bind("e", lambda e: self.flag())
        self.win.bind("E", lambda e: self.flag())
        self.win.bind("<Delete>", lambda e: self.delete())
        self.win.bind("a", lambda e: self.prev())
        self.win.bind("d", lambda e: self.next())
        self.win.bind("<Left>", lambda e: self.prev())
        self.win.bind("<Right>", lambda e: self.next())
        self.win.bind("<Escape>", lambda e: self.close())
        self.win.focus_set()

        self.tk_img = None
        self.show()

    # ---------------------------------------------------------------- عرض

    def current(self) -> dict:
        return self.refs[self.pos]

    def show(self):
        r = self.current()
        if self.app.is_flagged(r):
            v = "⚠ موسوم كلاس خاطئ"
        elif self.app.is_verified(r):
            v = "✔ مراجَع"
        else:
            v = "⬜ غير مراجَع"
        self.win.title(f"مراجعة {self.pos + 1}/{len(self.refs)} — {r['stem']}")
        sus = self.app.auto_suspects.get(box_key(r["stem"], r["idx"]))
        hint = ""
        if sus and sus.get("cls") == r["cls"]:
            hint = (f"  |  🤖 النموذج يرى '{cls_name(sus['suggest'])}' "
                    f"({sus.get('conf', 0):.2f})")
        self.banner.set(f"[{self.pos + 1}/{len(self.refs)}]  {r['stem']}  |  "
                        f"الكلاس الحالي: {cls_name(r['cls'])}  |  "
                        f"{'💥 مدمّر' if r.get('dmg') else 'سليم'}  |  {v}{hint}")
        self.canvas.delete("all")

        path = self.app.image_index.get(r["stem"])
        if path is None:
            self.canvas.create_text(self.CANVAS_W // 2, self.CANVAS_H // 2,
                                    text=f"الصورة مفقودة محلياً:\n{r['stem']}",
                                    fill="#fff", font=("Segoe UI", 14))
            return
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
        except OSError as e:
            self.canvas.create_text(self.CANVAS_W // 2, self.CANVAS_H // 2,
                                    text=f"تعذّر فتح الصورة:\n{e}", fill="#fff")
            return

        cw = max(self.canvas.winfo_width(), self.CANVAS_W)
        ch = max(self.canvas.winfo_height(), self.CANVAS_H)
        W, H = im.size
        scale = min(cw / W, ch / H, 2.0)
        disp = im.resize((int(W * scale), int(H * scale)))

        draw = ImageDraw.Draw(disp)
        color = CLASS_COLORS[r["cls"]] if 0 <= r["cls"] < len(CLASSES) else "#ff0000"
        # بقية صناديق الصورة بخط رفيع باهت — للسياق
        label_path = label_file_for(self.app.repo, r["stem"])
        if label_path is not None:
            for j, other in enumerate(parse_label_file(label_path)):
                if j == r["idx"]:
                    continue
                self._draw_box(draw, other, disp.size, width=1, outline="#888888")
        self._draw_box(draw, r, disp.size, width=4, outline=color)

        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_img)

    @staticmethod
    def _draw_box(draw: ImageDraw.ImageDraw, b: dict, size: tuple[int, int],
                  width: int, outline: str):
        W, H = size
        if b.get("pts"):
            pts = [(x * W, y * H) for x, y in b["pts"]]
            draw.polygon(pts, outline=outline, width=width)
        else:
            x0, y0, x1, y1 = b["rect"]
            draw.rectangle((x0 * W, y0 * H, x1 * W, y1 * H),
                           outline=outline, width=width)

    # ---------------------------------------------------------------- أفعال

    def confirm(self):
        self.app.mark_verified(self.current())
        self.next(auto=True)

    def flag(self):
        self.app.apply_flag(self.current())
        self.next(auto=True)

    def reassign(self, new_cls: int):
        self.app.apply_class_change(self.current(), new_cls)
        self.next(auto=True)

    def delete(self):
        if self.app.apply_delete(self.current()):
            del self.refs[self.pos]
            if not self.refs:
                self.close()
                return
            self.pos = min(self.pos, len(self.refs) - 1)
            self.show()
            self.app.refresh_page(keep_page=True)

    def next(self, auto: bool = False):
        if self.pos + 1 < len(self.refs):
            self.pos += 1
            self.show()
        elif auto:
            self.close()
            messagebox.showinfo("انتهى", "راجعت آخر صندوق في هذا الكلاس 🎉")

    def prev(self):
        if self.pos > 0:
            self.pos -= 1
            self.show()

    def close(self):
        self.alive = False
        self.app.refresh_page(keep_page=True)
        self.win.destroy()


# ----------------------------------------------------------------------------
# التشغيل
# ----------------------------------------------------------------------------

def main() -> int:
    # طرفية Windows الافتراضية cp1252 لا تطبع العربية
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="أداة مراجعة صحة الكلاسات")
    ap.add_argument("--repo", type=Path, default=Path("."),
                    help="جذر الريبو (افتراضياً المجلد الحالي)")
    ap.add_argument("--user", default=getpass.getuser(), help="اسم المراجِع")
    ap.add_argument("--scan", action="store_true",
                    help="فحص سريع في الطرفية (توزيع الكلاسات + خارج النطاق) بلا واجهة")
    ap.add_argument("--auto", action="store_true",
                    help="فحص تلقائي بـ YOLO-World — إشارة فقط: يذكر الصور المشتبه "
                         "بها في التقرير وملف reports/auto_class_suspects.json "
                         "دون أي تعديل تلقائي")
    ap.add_argument("--model", default="yolov8l-worldv2.pt",
                    help="نموذج YOLO-World (يُنزَّل تلقائياً أول مرة)")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="أدنى ثقة لاعتبار تنبؤ الكلاس المختلف مشتبهاً به")
    ap.add_argument("--iou", type=float, default=0.45,
                    help="أدنى IoU لمطابقة تنبؤ النموذج مع صندوق الـ GT")
    ap.add_argument("--device", default=None, help="مثل cuda:0 أو cpu (اختياري)")
    ap.add_argument("--limit", type=int, default=None,
                    help="فحص أول N صورة فقط (للتجربة)")
    args = ap.parse_args()

    repo = args.repo.resolve()
    if not (repo / LABELS_DIR).exists() and not (repo / LABELS_OBB_DIR).exists():
        print(f"لا يوجد data/annotations داخل: {repo}", file=sys.stderr)
        return 1

    if args.scan:
        return cli_scan(repo)
    if args.auto:
        return auto_check(repo, args.model, args.conf, args.iou,
                          args.device, args.limit)

    root = tk.Tk()
    ClassCheckApp(root, repo, args.user)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

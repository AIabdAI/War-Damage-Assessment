#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
War-Damage-Assessment — Team Annotation Tool
=============================================
أداة أنوتيشن جماعية لمشروع تقييم أضرار المباني.

- ترسم Bounding Boxes عادية (YOLO) ومدوّرة (YOLO-OBB).
- تدير الأقفال (Locks) بين أعضاء الفريق عبر processed_log.json المتزامن مع GitHub.
- تحفظ الـ labels في data/annotations (المُدار عبر DVC).
- Idempotency عبر MD5 + سجل الصور المعالجة.
- تصدير إحصائيات YAML وتقرير Markdown مع فتح Pull Request.

الاستخدام:
    python tools/annotation_tool.py            # من جذر الريبو
    python tools/annotation_tool.py --repo /path/to/War-Damage-Assessment

المتطلبات:
    pip install pillow pyyaml
"""

from __future__ import annotations

import argparse
import copy
import getpass
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

# ----------------------------------------------------------------------------
# الإعدادات العامة — متوافقة مع بنية الريبو الحالية
# ----------------------------------------------------------------------------

CLASSES = [
    "Brick_Wall", "Column", "Staircase", "Floor_Tiles", "Ceiling",
    "Beam", "Kitchen_Countertop", "Sink", "Wall_Cabinet", "Chair",
    "Floor_Cabinet", "Bathtub", "Water_Faucet", "Window", "Fire_Extinguisher",
    "Roof", "Door", "Air_Conditioner", "Window_Sill", "Light_Fixture",
]

CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
    "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#ff8c00", "#00bfff",
]

# اختصارات الكيبورد: 1..9,0 لأول عشرة، q..p للعشرة التالية
CLASS_KEYS = list("1234567890qwertyuiop")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

LOCK_HOURS = 2            # مدة صلاحية القفل
HANDLE_SIZE = 5           # نصف حجم مقبض التحجيم بالبكسل (على الشاشة)
ROT_HANDLE_DIST = 28      # بُعد مقبض التدوير عن الحافة العلوية
MIN_BOX_SIZE = 6          # أصغر حجم مسموح للـ box بالبكسل (إحداثيات الصورة)

RAW_DIR = Path("data/raw")
ANNOT_DIR = Path("data/annotations")
LABELS_DIR = ANNOT_DIR / "labels"          # YOLO عادي: cls cx cy w h
LABELS_OBB_DIR = ANNOT_DIR / "labels_obb"  # YOLO-OBB: cls x1 y1 x2 y2 x3 y3 x4 y4
REPORTS_DIR = Path("reports")
LOG_FILE = Path("processed_log.json")
USER_CFG = Path.home() / ".wda_annotator.json"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# هندسة الصناديق (عادية + مدوّرة)
# box = dict(cls, cx, cy, w, h, angle)  — الإحداثيات ببكسل الصورة، angle بالدرجات
# angle == 0  →  box عادي يُحفظ بتنسيق YOLO القياسي
# angle != 0  →  box مدوّر يُحفظ بتنسيق YOLO-OBB (أربع نقاط)
# ----------------------------------------------------------------------------

def box_corners(b: dict) -> list[tuple[float, float]]:
    """أركان الصندوق الأربعة بالترتيب: أعلى-يسار، أعلى-يمين، أسفل-يمين، أسفل-يسار."""
    a = math.radians(b["angle"])
    ca, sa = math.cos(a), math.sin(a)
    hw, hh = b["w"] / 2.0, b["h"] / 2.0
    pts = []
    for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        pts.append((b["cx"] + dx * ca - dy * sa, b["cy"] + dx * sa + dy * ca))
    return pts


def to_local(b: dict, x: float, y: float) -> tuple[float, float]:
    """تحويل نقطة من إحداثيات الصورة إلى الإطار المحلي للصندوق."""
    a = math.radians(b["angle"])
    ca, sa = math.cos(a), math.sin(a)
    dx, dy = x - b["cx"], y - b["cy"]
    return dx * ca + dy * sa, -dx * sa + dy * ca


def point_in_box(b: dict, x: float, y: float, pad: float = 0.0) -> bool:
    lx, ly = to_local(b, x, y)
    return abs(lx) <= b["w"] / 2 + pad and abs(ly) <= b["h"] / 2 + pad


def enclosing_rect(b: dict) -> tuple[float, float, float, float]:
    """أصغر مستطيل محاذٍ للمحاور يحتوي الصندوق (لأجل تنسيق YOLO القياسي)."""
    xs, ys = zip(*box_corners(b))
    return min(xs), min(ys), max(xs), max(ys)


def corners_to_box(pts: list[tuple[float, float]], cls: int) -> dict:
    """إعادة بناء (cx,cy,w,h,angle) من أربع نقاط OBB."""
    (x1, y1), (x2, y2), (x3, y3), _ = pts
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    w = math.hypot(x2 - x1, y2 - y1)
    h = math.hypot(x3 - x2, y3 - y2)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return {"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h, "angle": angle}


# ----------------------------------------------------------------------------
# قراءة/كتابة labels بتنسيق YOLO
#   نحفظ ملفين لكل صورة:
#   - labels/<stem>.txt      : كل الصناديق كـ YOLO قياسي (المدوّرة تُحوَّل إلى
#                              المستطيل المحيط بها) — جاهز لتدريب YOLO detect.
#   - labels_obb/<stem>.txt  : كل الصناديق كأربع نقاط (المصدر الكامل، بدون فقد
#                              معلومة الدوران) — جاهز لتدريب YOLO-OBB.
#   عند التحميل نقرأ labels_obb إن وُجد (يحوي كل شيء)، وإلا labels العادي.
# ----------------------------------------------------------------------------

def save_labels(stem: str, boxes: list[dict], img_w: int, img_h: int, repo: Path) -> None:
    def nx(v): return min(max(v / img_w, 0.0), 1.0)
    def ny(v): return min(max(v / img_h, 0.0), 1.0)

    (repo / LABELS_DIR).mkdir(parents=True, exist_ok=True)
    (repo / LABELS_OBB_DIR).mkdir(parents=True, exist_ok=True)

    std_lines, obb_lines = [], []
    for b in boxes:
        x0, y0, x1, y1 = enclosing_rect(b)
        cx, cy = nx((x0 + x1) / 2), ny((y0 + y1) / 2)
        w, h = nx(x1 - x0), ny(y1 - y0)
        std_lines.append(f"{b['cls']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        pts = box_corners(b)
        flat = " ".join(f"{nx(px):.6f} {ny(py):.6f}" for px, py in pts)
        obb_lines.append(f"{b['cls']} {flat}")

    (repo / LABELS_DIR / f"{stem}.txt").write_text("\n".join(std_lines) + ("\n" if std_lines else ""), encoding="utf-8")
    (repo / LABELS_OBB_DIR / f"{stem}.txt").write_text("\n".join(obb_lines) + ("\n" if obb_lines else ""), encoding="utf-8")


def load_labels(stem: str, img_w: int, img_h: int, repo: Path) -> list[dict]:
    obb = repo / LABELS_OBB_DIR / f"{stem}.txt"
    std = repo / LABELS_DIR / f"{stem}.txt"
    boxes: list[dict] = []
    if obb.exists():
        for line in obb.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 9:
                continue
            cls = int(float(parts[0]))
            vals = list(map(float, parts[1:]))
            pts = [(vals[i] * img_w, vals[i + 1] * img_h) for i in range(0, 8, 2)]
            b = corners_to_box(pts, cls)
            if abs(b["angle"]) < 0.05:
                b["angle"] = 0.0
            boxes.append(b)
    elif std.exists():
        for line in std.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
            boxes.append({"cls": cls, "cx": cx * img_w, "cy": cy * img_h,
                          "w": w * img_w, "h": h * img_h, "angle": 0.0})
    return boxes


# ----------------------------------------------------------------------------
# سجل المعالجة والأقفال — نفس مخطط processed_log.json الموجود في الريبو
# ----------------------------------------------------------------------------

def empty_log() -> dict:
    return {"annotated": {}, "augmented": {}, "locks": {}}


def read_log(path: Path) -> dict:
    if not path.exists():
        return empty_log()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty_log()
    for key in ("annotated", "augmented", "locks"):
        data.setdefault(key, {})
    return data


def write_log(path: Path, log: dict) -> None:
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def prune_expired_locks(log: dict) -> None:
    now = utcnow()
    dead = []
    for stem, lk in log["locks"].items():
        try:
            if parse_iso(lk["expires_at"]) <= now:
                dead.append(stem)
        except (KeyError, ValueError):
            dead.append(stem)
    for stem in dead:
        del log["locks"][stem]


def merge_logs(local: dict, remote: dict) -> dict:
    """دمج نسختين من السجل (محلية + من GitHub) بدون فقد بيانات أي مطوّر.

    - annotated / augmented : الاتحاد، وعند التعارض يفوز الأحدث زمنياً.
    - locks : الاتحاد بعد حذف المنتهي، وعند تعارض قفلين على نفس الصورة
      يفوز مَن حجز أولاً (الأقدم locked_at).
    """
    merged = empty_log()
    for section in ("annotated", "augmented"):
        merged[section] = dict(remote.get(section, {}))
        for stem, entry in local.get(section, {}).items():
            other = merged[section].get(stem)
            if other is None:
                merged[section][stem] = entry
            else:
                t_l = entry.get("timestamp", "")
                t_r = other.get("timestamp", "")
                merged[section][stem] = entry if t_l >= t_r else other

    locks = dict(remote.get("locks", {}))
    for stem, lk in local.get("locks", {}).items():
        other = locks.get(stem)
        if other is None:
            locks[stem] = lk
        else:
            try:
                first = lk if parse_iso(lk["locked_at"]) <= parse_iso(other["locked_at"]) else other
            except (KeyError, ValueError):
                first = other
            locks[stem] = first
    merged["locks"] = locks
    prune_expired_locks(merged)
    return merged


class LockManager:
    """إدارة الأقفال والسجل مع المزامنة عبر git."""

    def __init__(self, repo: Path, user: str):
        self.repo = repo
        self.user = user
        self.log_path = repo / LOG_FILE
        self.log = read_log(self.log_path)
        prune_expired_locks(self.log)

    # ---------------- git helpers ----------------

    def _git(self, *args: str, check: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=self.repo, check=check,
            capture_output=True, text=True,
        )

    def git_available(self) -> bool:
        return self._git("rev-parse", "--is-inside-work-tree").returncode == 0

    def _remote_log(self) -> dict:
        """قراءة processed_log.json من آخر نسخة على origin بدون لمس ملفات العمل."""
        branch = self._git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or "master"
        self._git("fetch", "origin", branch)
        show = self._git("show", f"origin/{branch}:{LOG_FILE.as_posix()}")
        if show.returncode != 0:
            return empty_log()
        try:
            data = json.loads(show.stdout)
            for key in ("annotated", "augmented", "locks"):
                data.setdefault(key, {})
            return data
        except json.JSONDecodeError:
            return empty_log()

    def sync(self, message: str, extra_paths: list[Path] | None = None) -> tuple[bool, str]:
        """مزامنة السجل مع GitHub: fetch → merge → commit → push (مع إعادة محاولة).

        تُستدعى عند كل حفظ. الدمج يتم على مستوى البيانات (وليس نص الملف)
        لذا لا تحدث تعارضات git على processed_log.json.
        """
        if not self.git_available():
            write_log(self.log_path, self.log)
            return False, "المجلد ليس مستودع git — تم الحفظ محلياً فقط."

        branch = self._git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or "master"
        last_err = ""
        for _attempt in range(4):
            # 1) جلب أحدث نسخة من السجل من origin ودمجها على مستوى البيانات
            remote = self._remote_log()
            self.log = merge_logs(self.log, remote)
            write_log(self.log_path, self.log)

            add_paths = [str(LOG_FILE)] + [str(p) for p in (extra_paths or [])]
            self._git("add", *add_paths)
            commit = self._git("commit", "-m", message)
            if commit.returncode != 0 and "nothing to commit" not in (commit.stdout + commit.stderr):
                last_err = (commit.stderr or commit.stdout).strip()

            # 2) دمج تغييرات origin (merge وليس rebase — أسهل في حل التعارض)
            pull = self._git("pull", "--no-rebase", "--no-edit", "--autostash",
                             "origin", branch)
            if pull.returncode != 0:
                # تعارض دمج — غالباً على processed_log.json فقط
                merge_head = self._git("show", f"MERGE_HEAD:{LOG_FILE.as_posix()}")
                try:
                    their_log = json.loads(merge_head.stdout) if merge_head.returncode == 0 else empty_log()
                    for key in ("annotated", "augmented", "locks"):
                        their_log.setdefault(key, {})
                except json.JSONDecodeError:
                    their_log = empty_log()
                self.log = merge_logs(self.log, their_log)
                write_log(self.log_path, self.log)
                self._git("add", str(LOG_FILE))
                # أي ملفات أخرى متعارضة (نادرة بفضل نظام الأقفال): نأخذ نسخة origin
                conflicted = self._git("diff", "--name-only", "--diff-filter=U").stdout.split()
                for f in conflicted:
                    if f != LOG_FILE.as_posix():
                        self._git("checkout", "--theirs", f)
                        self._git("add", f)
                finish = self._git("commit", "--no-edit")
                if finish.returncode != 0:
                    self._git("merge", "--abort")
                    last_err = (pull.stderr or pull.stdout).strip()
                    continue

            # 3) الرفع — وإن سبقنا أحدهم بين الجلب والرفع نعيد المحاولة
            push = self._git("push", "origin", branch)
            if push.returncode == 0:
                return True, "تمت المزامنة مع GitHub بنجاح."
            last_err = push.stderr.strip()
        return False, f"تعذّرت المزامنة مع GitHub (تم الحفظ محلياً):\n{last_err[-400:]}"

    # ---------------- locks ----------------

    def lock_owner(self, stem: str) -> str | None:
        lk = self.log["locks"].get(stem)
        if not lk:
            return None
        try:
            if parse_iso(lk["expires_at"]) <= utcnow():
                del self.log["locks"][stem]
                return None
        except (KeyError, ValueError):
            del self.log["locks"][stem]
            return None
        return lk.get("locked_by")

    def acquire(self, stem: str) -> bool:
        owner = self.lock_owner(stem)
        if owner is not None and owner != self.user:
            return False
        now = utcnow()
        self.log["locks"][stem] = {
            "locked_by": self.user,
            "locked_at": iso(now),
            "expires_at": iso(now + timedelta(hours=LOCK_HOURS)),
        }
        return True

    def release(self, stem: str) -> None:
        lk = self.log["locks"].get(stem)
        if lk and lk.get("locked_by") == self.user:
            del self.log["locks"][stem]

    def release_all_mine(self) -> None:
        for stem in [s for s, lk in self.log["locks"].items() if lk.get("locked_by") == self.user]:
            del self.log["locks"][stem]

    # ---------------- annotated registry ----------------

    def is_annotated(self, stem: str, img_path: Path) -> bool:
        """Idempotency: الصورة معالَجة إذا سُجّلت ولم يتغيّر محتواها (MD5)."""
        entry = self.log["annotated"].get(stem)
        if not entry:
            return False
        try:
            return entry.get("hash") == file_md5(img_path)
        except OSError:
            return False

    def record_annotation(self, stem: str, img_path: Path, boxes: list[dict]) -> None:
        class_boxes = {name: 0 for name in CLASSES}
        for b in boxes:
            if 0 <= b["cls"] < len(CLASSES):
                class_boxes[CLASSES[b["cls"]]] += 1
        self.log["annotated"][stem] = {
            "hash": file_md5(img_path),
            "timestamp": iso(utcnow()),
            "boxes": len(boxes),
            "annotated_by": self.user,
            "class_boxes": class_boxes,
        }

# ----------------------------------------------------------------------------
# الإحصائيات والتقارير
# ----------------------------------------------------------------------------

def build_stats(log: dict) -> dict:
    annotated = log.get("annotated", {})
    totals = {name: 0 for name in CLASSES}
    per_dev: dict[str, int] = {}
    total_boxes = 0
    for entry in annotated.values():
        total_boxes += entry.get("boxes", 0)
        dev = entry.get("annotated_by", "unknown")
        per_dev[dev] = per_dev.get(dev, 0) + 1
        for cls, n in entry.get("class_boxes", {}).items():
            if cls in totals:
                totals[cls] += n
    return {
        "generated_at": iso(utcnow()),
        "total_images_annotated": len(annotated),
        "total_boxes": total_boxes,
        "images_per_developer": dict(sorted(per_dev.items(), key=lambda kv: -kv[1])),
        "boxes_per_class_total": totals,
        "per_image": {
            stem: {
                "annotated_by": e.get("annotated_by"),
                "timestamp": e.get("timestamp"),
                "boxes": e.get("boxes", 0),
                "class_boxes": {k: v for k, v in e.get("class_boxes", {}).items() if v},
            }
            for stem, e in sorted(annotated.items())
        },
    }


def export_stats_yaml(log: dict, out_path: Path) -> Path:
    stats = build_stats(log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        out_path.write_text(
            yaml.safe_dump(stats, allow_unicode=True, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
    else:  # fallback بسيط إذا لم تتوفر PyYAML
        out_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def build_markdown_report(log: dict, total_raw_images: int) -> str:
    stats = build_stats(log)
    done = stats["total_images_annotated"]
    pct = (100.0 * done / total_raw_images) if total_raw_images else 0.0
    lines = [
        "# Annotation Progress Report — War-Damage-Assessment",
        "",
        f"- **Generated:** {stats['generated_at']}",
        f"- **Images annotated:** {done} / {total_raw_images} ({pct:.1f}%)",
        f"- **Total bounding boxes:** {stats['total_boxes']}",
        "",
        "## Images per developer",
        "",
        "| Developer | Images |",
        "|---|---|",
    ]
    for dev, n in stats["images_per_developer"].items():
        lines.append(f"| {dev} | {n} |")
    lines += [
        "",
        "## Boxes per class (total)",
        "",
        "| Class | Boxes |",
        "|---|---|",
    ]
    for cls, n in sorted(stats["boxes_per_class_total"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {cls} | {n} |")
    active = {s: lk for s, lk in log.get("locks", {}).items()}
    lines += ["", "## Active locks", ""]
    if active:
        lines += ["| Image | Locked by | Expires |", "|---|---|---|"]
        for stem, lk in sorted(active.items()):
            lines.append(f"| {stem} | {lk.get('locked_by')} | {lk.get('expires_at')} |")
    else:
        lines.append("_No active locks._")
    lines.append("")
    return "\n".join(lines)


def publish_report_pr(lm: LockManager, report_md: str, stats_path: Path) -> str:
    """إنشاء فرع + رفع التقرير + فتح PR (عبر gh CLI إن وُجد، وإلا رابط المقارنة)."""
    repo = lm.repo
    reports = repo / REPORTS_DIR
    reports.mkdir(parents=True, exist_ok=True)
    stamp = utcnow().strftime("%Y%m%d-%H%M")
    report_path = reports / "annotation_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    base = lm._git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or "master"
    branch = f"report/annotation-{stamp}-{lm.user.lower().replace(' ', '-')}"

    lm._git("checkout", "-b", branch)
    lm._git("add", str(REPORTS_DIR / "annotation_report.md"), str(stats_path.relative_to(repo)))
    lm._git("commit", "-m", f"docs(report): annotation progress report {stamp} by {lm.user}")
    push = lm._git("push", "-u", "origin", branch)
    lm._git("checkout", base)
    if push.returncode != 0:
        return f"تعذّر رفع الفرع {branch}:\n{push.stderr.strip()[-300:]}"

    gh = subprocess.run(
        ["gh", "pr", "create", "--base", base, "--head", branch,
         "--title", f"Annotation progress report — {stamp}",
         "--body", "Automated report generated by the team annotation tool."],
        cwd=repo, capture_output=True, text=True,
    )
    if gh.returncode == 0:
        return f"تم فتح الـ PR بنجاح:\n{gh.stdout.strip()}"
    url = lm._git("remote", "get-url", "origin").stdout.strip()
    if url.endswith(".git"):
        url = url[:-4]
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split(":", 1)[1]
    return (f"تم رفع الفرع {branch}.\n"
            f"gh CLI غير متوفر — افتح الـ PR يدوياً من:\n{url}/compare/{base}...{branch}?expand=1")

# ----------------------------------------------------------------------------
# واجهة المستخدم الرسومية (Tkinter)
# ----------------------------------------------------------------------------

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError:
    print("مطلوب تثبيت Pillow:  pip install pillow", file=sys.stderr)
    raise


class AnnotationApp:
    def __init__(self, root: tk.Tk, repo: Path, user: str):
        self.root = root
        self.repo = repo
        self.user = user
        self.lm = LockManager(repo, user)

        self.images: list[Path] = []
        self.idx = -1
        self.img: Image.Image | None = None
        self.tk_img = None
        self.img_w = self.img_h = 0

        self.boxes: list[dict] = []
        self.selected: int | None = None
        self.clipboard: list[dict] = []
        self.dirty = False
        self.current_class = 0

        # حالة التفاعل بالماوس
        self.mode = None          # draw | move | resize | rotate | None
        self.drag_start = (0, 0)
        self.drag_box0: dict | None = None
        self.resize_handle = -1
        self.rotate_offset = 0.0

        # عرض الصورة: تكبير + إزاحة
        self.scale = 1.0
        self.off_x = self.off_y = 0.0

        self._build_ui()
        self._bind_keys()

        raw = repo / RAW_DIR
        if raw.exists():
            self.load_folder(raw)

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        self.root.title(f"WDA Annotation Tool — {self.user}")
        self.root.geometry("1360x860")

        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=4, pady=3)
        tk.Button(top, text="📁 فتح مجلد", command=self.choose_folder).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="⟨ السابق (A)", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="التالي (D) ⟩", command=self.next_image).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="⏭ التالي غير المعالَج", command=self.next_unannotated).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="💾 حفظ + مزامنة (Ctrl+S)", bg="#d1f0d1",
                  command=self.save_current).pack(side=tk.LEFT, padx=8)
        tk.Button(top, text="🗑 حذف المحدد (Del)", command=self.delete_selected).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="🧹 مسح الكل", command=self.clear_all).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="📊 الإحصائيات", command=self.show_stats).pack(side=tk.RIGHT, padx=2)
        tk.Button(top, text="📄 تقرير + PR", command=self.report_pr).pack(side=tk.RIGHT, padx=2)
        tk.Button(top, text="🟡 YAML", command=self.export_yaml).pack(side=tk.RIGHT, padx=2)

        self.status = tk.StringVar(value="اختر مجلد الصور للبدء…")
        tk.Label(self.root, textvariable=self.status, anchor="w",
                 bg="#222", fg="#eee").pack(side=tk.BOTTOM, fill=tk.X)

        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # لوحة الكلاسات على اليمين
        side = tk.Frame(main, width=250)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=4, pady=4)
        side.pack_propagate(False)

        tk.Label(side, text="الكلاس الحالي:").pack(anchor="w")
        self.class_var = tk.StringVar(value=CLASSES[0])
        combo = ttk.Combobox(side, textvariable=self.class_var, values=CLASSES, state="readonly")
        combo.pack(fill=tk.X, pady=2)
        combo.bind("<<ComboboxSelected>>",
                   lambda e: self.set_class(CLASSES.index(self.class_var.get())))

        grid = tk.Frame(side)
        grid.pack(fill=tk.X, pady=4)
        self.class_btns = []
        for i, name in enumerate(CLASSES):
            fg = "#000" if i in (7, 12, 14, 16) else "#fff"
            b = tk.Button(grid, text=f"{CLASS_KEYS[i]}  {name}", bg=CLASS_COLORS[i], fg=fg,
                          anchor="w", relief=tk.RAISED, font=("TkDefaultFont", 8),
                          command=lambda i=i: self.set_class(i))
            b.grid(row=i % 10, column=i // 10, sticky="ew", padx=1, pady=1)
            self.class_btns.append(b)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        tk.Label(side, text="صناديق هذه الصورة:").pack(anchor="w", pady=(8, 0))
        self.box_list = tk.Listbox(side, height=12)
        self.box_list.pack(fill=tk.BOTH, expand=True)
        self.box_list.bind("<<ListboxSelect>>", self._on_list_select)

        help_txt = ("سحب: رسم box جديد • سحب داخل box: تحريك\n"
                    "المقابض المربعة: تغيير الحجم • الدائرة: تدوير\n"
                    "Ctrl+C/V نسخ/لصق • Del حذف • عجلة الماوس: تكبير")
        tk.Label(side, text=help_txt, justify="right", fg="#555",
                 font=("TkDefaultFont", 8)).pack(anchor="e", pady=4)

        # الكانفاس
        self.canvas = tk.Canvas(main, bg="#333", cursor="tcross")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<MouseWheel>", self.on_wheel)          # Windows/macOS
        self.canvas.bind("<Button-4>", lambda e: self.on_wheel(e, 1))   # Linux
        self.canvas.bind("<Button-5>", lambda e: self.on_wheel(e, -1))
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    def _bind_keys(self):
        r = self.root
        for i, key in enumerate(CLASS_KEYS):
            r.bind(key, lambda e, i=i: self.set_class(i))
            r.bind(key.upper(), lambda e, i=i: self.set_class(i))
        r.bind("<Control-s>", lambda e: self.save_current())
        r.bind("<Control-c>", lambda e: self.copy_selected())
        r.bind("<Control-v>", lambda e: self.paste_clipboard())
        r.bind("<Delete>", lambda e: self.delete_selected())
        r.bind("<Escape>", lambda e: self.select_box(None))
        r.bind("a", lambda e: self.prev_image())
        r.bind("d", lambda e: self.next_image())
        r.bind("<Left>", lambda e: self.prev_image())
        r.bind("<Right>", lambda e: self.next_image())
        r.protocol("WM_DELETE_WINDOW", self.on_close)

    def set_class(self, i: int):
        self.current_class = i
        self.class_var.set(CLASSES[i])
        for j, b in enumerate(self.class_btns):
            b.configure(relief=tk.SUNKEN if j == i else tk.RAISED)
        if self.selected is not None:
            self.boxes[self.selected]["cls"] = i
            self.dirty = True
            self.redraw()
        self._update_status()

    # ------------------------------------------------------- التنقل والصور --

    def choose_folder(self):
        d = filedialog.askdirectory(initialdir=str(self.repo / RAW_DIR),
                                    title="اختر مجلد الصور (data/raw)")
        if d:
            self.load_folder(Path(d))

    def load_folder(self, folder: Path):
        self.images = sorted(p for p in folder.iterdir()
                             if p.suffix.lower() in IMG_EXTS)
        if not self.images:
            messagebox.showwarning("لا صور", f"لا توجد صور في:\n{folder}")
            return
        self.idx = -1
        self.next_unannotated(from_start=True)

    def _stem(self, i: int | None = None) -> str:
        i = self.idx if i is None else i
        return self.images[i].stem

    def _confirm_leave(self) -> bool:
        if not self.dirty:
            return True
        ans = messagebox.askyesnocancel("تغييرات غير محفوظة",
                                        "توجد تغييرات غير محفوظة. حفظ قبل المتابعة؟")
        if ans is None:
            return False
        if ans:
            return self.save_current()
        self.dirty = False
        return True

    def open_index(self, i: int, force: bool = False):
        if not (0 <= i < len(self.images)):
            return
        if not self._confirm_leave():
            return
        stem = self._stem(i)
        owner = self.lm.lock_owner(stem)
        if owner is not None and owner != self.user and not force:
            messagebox.showwarning("صورة محجوزة",
                                   f"الصورة {stem} محجوزة حالياً بواسطة: {owner}\n"
                                   f"(ينتهي القفل تلقائياً بعد ساعتين من حجزه)")
            return
        # تحرير قفل الصورة السابقة عند مغادرتها
        if 0 <= self.idx < len(self.images) and self.idx != i:
            self.lm.release(self._stem())

        self.idx = i
        path = self.images[i]
        try:
            self.img = Image.open(path).convert("RGB")
        except OSError as e:
            messagebox.showerror("خطأ", f"تعذّر فتح الصورة:\n{e}")
            return
        self.img_w, self.img_h = self.img.size
        self.boxes = load_labels(stem, self.img_w, self.img_h, self.repo)
        self.selected = None
        self.dirty = False
        self.fit_view()

        # حجز الصورة الحالية + حجز التالية غير المعالَجة استباقياً
        self.lm.acquire(stem)
        nxt = self._find_next_free(i + 1)
        if nxt is not None:
            self.lm.acquire(self._stem(nxt))
        write_log(self.lm.log_path, self.lm.log)

        self._refresh_box_list()
        self.redraw()
        self._update_status()

    def _find_next_free(self, start: int) -> int | None:
        for j in range(start, len(self.images)):
            stem = self._stem(j)
            if self.lm.is_annotated(stem, self.images[j]):
                continue
            owner = self.lm.lock_owner(stem)
            if owner is None or owner == self.user:
                return j
        return None

    def next_image(self):
        self.open_index(self.idx + 1)

    def prev_image(self):
        self.open_index(self.idx - 1)

    def next_unannotated(self, from_start: bool = False):
        start = 0 if from_start else self.idx + 1
        j = self._find_next_free(start)
        if j is None:
            messagebox.showinfo("انتهى", "لا توجد صور متاحة غير معالَجة 🎉")
            return
        self.open_index(j)

    def _update_status(self):
        if self.idx < 0:
            return
        stem = self._stem()
        annotated = "✅ معالَجة" if self.lm.is_annotated(stem, self.images[self.idx]) else "⬜ غير معالَجة"
        nxt = self._find_next_free(self.idx + 1)
        reserved = self._stem(nxt) if nxt is not None else "—"
        self.status.set(
            f"[{self.idx + 1}/{len(self.images)}] {stem}  |  {annotated}  |  "
            f"صناديق: {len(self.boxes)}  |  الكلاس: {CLASSES[self.current_class]}  |  "
            f"محجوز لك أيضاً: {reserved}  |  المستخدم: {self.user}"
        )

    # ------------------------------------------------ إحداثيات العرض/الصورة --

    def fit_view(self):
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        if self.img is None:
            return
        self.scale = min(cw / self.img_w, ch / self.img_h, 4.0)
        self.off_x = (cw - self.img_w * self.scale) / 2
        self.off_y = (ch - self.img_h * self.scale) / 2

    def to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.off_x, y * self.scale + self.off_y

    def to_image(self, cx: float, cy: float) -> tuple[float, float]:
        return (cx - self.off_x) / self.scale, (cy - self.off_y) / self.scale

    def on_wheel(self, event, direction: int | None = None):
        if self.img is None:
            return
        d = direction if direction is not None else (1 if event.delta > 0 else -1)
        factor = 1.15 if d > 0 else 1 / 1.15
        ix, iy = self.to_image(event.x, event.y)
        self.scale = max(0.05, min(self.scale * factor, 12.0))
        self.off_x = event.x - ix * self.scale
        self.off_y = event.y - iy * self.scale
        self.redraw()

    def on_pan_start(self, event):
        self._pan0 = (event.x, event.y, self.off_x, self.off_y)

    def on_pan_drag(self, event):
        x0, y0, ox, oy = self._pan0
        self.off_x = ox + event.x - x0
        self.off_y = oy + event.y - y0
        self.redraw()

    # -------------------------------------------------- التفاعل مع الصناديق --

    def _hit_test(self, ix: float, iy: float):
        """يرجع (نوع, بيانات): rotate / resize(h) / move(idx) / None."""
        tol = HANDLE_SIZE * 2 / self.scale
        if self.selected is not None:
            b = self.boxes[self.selected]
            # مقبض التدوير (فوق منتصف الحافة العلوية)
            rx, ry = self._rotate_handle_pos(b)
            if math.hypot(ix - rx, iy - ry) <= tol:
                return "rotate", None
            # مقابض التحجيم الثمانية (في الإطار المحلي)
            lx, ly = to_local(b, ix, iy)
            hw, hh = b["w"] / 2, b["h"] / 2
            handles = [(-hw, -hh), (0, -hh), (hw, -hh), (hw, 0),
                       (hw, hh), (0, hh), (-hw, hh), (-hw, 0)]
            for h, (px, py) in enumerate(handles):
                if abs(lx - px) <= tol and abs(ly - py) <= tol:
                    return "resize", h
        # النقر داخل box (الأصغر مساحةً أولاً حتى يمكن اختيار الداخلي)
        candidates = [(b["w"] * b["h"], i) for i, b in enumerate(self.boxes)
                      if point_in_box(b, ix, iy)]
        if candidates:
            _, i = min(candidates)
            return "move", i
        return None, None

    def _rotate_handle_pos(self, b: dict) -> tuple[float, float]:
        a = math.radians(b["angle"])
        d = b["h"] / 2 + ROT_HANDLE_DIST / self.scale
        return b["cx"] + d * math.sin(a), b["cy"] - d * math.cos(a)

    def on_press(self, event):
        if self.img is None:
            return
        ix, iy = self.to_image(event.x, event.y)
        kind, data = self._hit_test(ix, iy)
        self.drag_start = (ix, iy)

        if kind == "rotate":
            b = self.boxes[self.selected]
            self.mode = "rotate"
            self.rotate_offset = math.degrees(
                math.atan2(iy - b["cy"], ix - b["cx"])) - b["angle"]
            self.drag_box0 = copy.deepcopy(b)
        elif kind == "resize":
            self.mode = "resize"
            self.resize_handle = data
            self.drag_box0 = copy.deepcopy(self.boxes[self.selected])
        elif kind == "move":
            self.select_box(data)
            self.mode = "move"
            self.drag_box0 = copy.deepcopy(self.boxes[data])
        else:
            # بدء رسم box جديد
            self.select_box(None)
            self.mode = "draw"
            self.boxes.append({"cls": self.current_class, "cx": ix, "cy": iy,
                               "w": 1.0, "h": 1.0, "angle": 0.0})
            self.select_box(len(self.boxes) - 1)

    def on_drag(self, event):
        if self.img is None or self.mode is None:
            return
        ix, iy = self.to_image(event.x, event.y)
        x0, y0 = self.drag_start

        if self.mode == "draw":
            b = self.boxes[self.selected]
            b["cx"], b["cy"] = (x0 + ix) / 2, (y0 + iy) / 2
            b["w"], b["h"] = abs(ix - x0), abs(iy - y0)
        elif self.mode == "move":
            b0, b = self.drag_box0, self.boxes[self.selected]
            b["cx"] = b0["cx"] + (ix - x0)
            b["cy"] = b0["cy"] + (iy - y0)
        elif self.mode == "rotate":
            b = self.boxes[self.selected]
            ang = math.degrees(math.atan2(iy - b["cy"], ix - b["cx"])) - self.rotate_offset
            # Shift = تثبيت على مضاعفات 15 درجة
            if event.state & 0x0001:
                ang = round(ang / 15.0) * 15.0
            b["angle"] = ((ang + 180) % 360) - 180
        elif self.mode == "resize":
            b0 = self.drag_box0
            b = self.boxes[self.selected]
            lx, ly = to_local(b0, ix, iy)
            hw, hh = b0["w"] / 2, b0["h"] / 2
            sx = [-1, 0, 1, 1, 1, 0, -1, -1][self.resize_handle]
            sy = [-1, -1, -1, 0, 1, 1, 1, 1][self.resize_handle]
            # الحافة/الركن المقابل يبقى ثابتاً
            if sx:
                new_w = max(MIN_BOX_SIZE, abs(lx - (-sx * hw)))
                cx_l = (lx + (-sx * hw)) / 2
            else:
                new_w, cx_l = b0["w"], 0.0
            if sy:
                new_h = max(MIN_BOX_SIZE, abs(ly - (-sy * hh)))
                cy_l = (ly + (-sy * hh)) / 2
            else:
                new_h, cy_l = b0["h"], 0.0
            a = math.radians(b0["angle"])
            ca, sa = math.cos(a), math.sin(a)
            b["w"], b["h"] = new_w, new_h
            b["cx"] = b0["cx"] + cx_l * ca - cy_l * sa
            b["cy"] = b0["cy"] + cx_l * sa + cy_l * ca

        self.dirty = True
        self.redraw()

    def on_release(self, event):
        if self.mode == "draw" and self.selected is not None:
            b = self.boxes[self.selected]
            if b["w"] < MIN_BOX_SIZE or b["h"] < MIN_BOX_SIZE:
                self.boxes.pop(self.selected)
                self.select_box(None)
        self.mode = None
        self.drag_box0 = None
        self._refresh_box_list()
        self.redraw()
        self._update_status()

    def select_box(self, i: int | None):
        self.selected = i
        if i is not None:
            self.set_class(self.boxes[i]["cls"])
            self.box_list.selection_clear(0, tk.END)
            if i < self.box_list.size():
                self.box_list.selection_set(i)
        self.redraw()

    def _on_list_select(self, _):
        sel = self.box_list.curselection()
        if sel:
            self.selected = sel[0]
            self.redraw()

    # -------------------------------------------------- نسخ / لصق / حذف --

    def copy_selected(self):
        if self.selected is None:
            if self.boxes:
                self.clipboard = copy.deepcopy(self.boxes)
                self.status.set(f"نُسخت كل الصناديق ({len(self.clipboard)}) — الصقها بـ Ctrl+V في أي صورة")
            return
        self.clipboard = [copy.deepcopy(self.boxes[self.selected])]
        self.status.set("نُسخ الصندوق المحدد — الصقه بـ Ctrl+V هنا أو في صورة أخرى")

    def paste_clipboard(self):
        if not self.clipboard or self.img is None:
            return
        offset = 12 / self.scale
        first_new = len(self.boxes)
        for b in self.clipboard:
            nb = copy.deepcopy(b)
            nb["cx"] = min(max(nb["cx"] + offset, 0), self.img_w)
            nb["cy"] = min(max(nb["cy"] + offset, 0), self.img_h)
            self.boxes.append(nb)
        self.dirty = True
        self.select_box(first_new)
        self._refresh_box_list()
        self.redraw()
        self._update_status()

    def delete_selected(self):
        if self.selected is None:
            return
        self.boxes.pop(self.selected)
        self.selected = None
        self.dirty = True
        self._refresh_box_list()
        self.redraw()
        self._update_status()

    def clear_all(self):
        if not self.boxes:
            return
        if messagebox.askyesno("مسح الكل", f"حذف كل الصناديق ({len(self.boxes)}) في هذه الصورة؟"):
            self.boxes.clear()
            self.selected = None
            self.dirty = True
            self._refresh_box_list()
            self.redraw()
            self._update_status()

    def _refresh_box_list(self):
        self.box_list.delete(0, tk.END)
        for b in self.boxes:
            rot = f" ∠{b['angle']:.0f}°" if abs(b["angle"]) > 0.05 else ""
            self.box_list.insert(tk.END, f"{CLASSES[b['cls']]}{rot}")

    # ------------------------------------------------------------- الرسم --

    def redraw(self):
        c = self.canvas
        c.delete("all")
        if self.img is None:
            return
        disp_w = max(1, int(self.img_w * self.scale))
        disp_h = max(1, int(self.img_h * self.scale))
        resized = self.img.resize((disp_w, disp_h), Image.BILINEAR)
        self.tk_img = ImageTk.PhotoImage(resized)
        c.create_image(self.off_x, self.off_y, image=self.tk_img, anchor="nw")

        for i, b in enumerate(self.boxes):
            color = CLASS_COLORS[b["cls"] % len(CLASS_COLORS)]
            pts = [self.to_canvas(x, y) for x, y in box_corners(b)]
            flat = [v for p in pts for v in p]
            width = 3 if i == self.selected else 2
            c.create_polygon(*flat, outline=color, fill="", width=width)
            lx, ly = pts[0]
            label = CLASSES[b["cls"]]
            c.create_rectangle(lx, ly - 16, lx + 7 * len(label) + 6, ly,
                               fill=color, outline=color)
            c.create_text(lx + 3, ly - 8, text=label, anchor="w",
                          fill="#fff", font=("TkDefaultFont", 8, "bold"))

            if i == self.selected:
                # مقابض التحجيم
                a = math.radians(b["angle"])
                ca, sa = math.cos(a), math.sin(a)
                hw, hh = b["w"] / 2, b["h"] / 2
                for px, py in [(-hw, -hh), (0, -hh), (hw, -hh), (hw, 0),
                               (hw, hh), (0, hh), (-hw, hh), (-hw, 0)]:
                    gx = b["cx"] + px * ca - py * sa
                    gy = b["cy"] + px * sa + py * ca
                    sxp, syp = self.to_canvas(gx, gy)
                    c.create_rectangle(sxp - HANDLE_SIZE, syp - HANDLE_SIZE,
                                       sxp + HANDLE_SIZE, syp + HANDLE_SIZE,
                                       fill="#fff", outline=color, width=2)
                # مقبض التدوير
                rx, ry = self._rotate_handle_pos(b)
                sxp, syp = self.to_canvas(rx, ry)
                tx, ty = self.to_canvas(*self._top_mid(b))
                c.create_line(tx, ty, sxp, syp, fill=color, dash=(3, 2))
                c.create_oval(sxp - 6, syp - 6, sxp + 6, syp + 6,
                              fill="#ffde59", outline=color, width=2)

    @staticmethod
    def _top_mid(b: dict) -> tuple[float, float]:
        a = math.radians(b["angle"])
        d = b["h"] / 2
        return b["cx"] + d * math.sin(a), b["cy"] - d * math.cos(a)

    # ---------------------------------------------------- الحفظ والمزامنة --

    def save_current(self) -> bool:
        if self.idx < 0 or self.img is None:
            return False
        stem = self._stem()
        img_path = self.images[self.idx]

        save_labels(stem, self.boxes, self.img_w, self.img_h, self.repo)
        self.lm.record_annotation(stem, img_path, self.boxes)
        self.lm.release(stem)  # انتهينا من هذه الصورة → تحرير قفلها
        self.dirty = False

        self.status.set("جارِ المزامنة مع GitHub…")
        self.root.update_idletasks()

        def do_sync():
            ok, msg = self.lm.sync(
                f"annotate({self.user}): {stem} — {len(self.boxes)} boxes")
            self.root.after(0, lambda: self._after_sync(ok, msg))

        threading.Thread(target=do_sync, daemon=True).start()
        return True

    def _after_sync(self, ok: bool, msg: str):
        self._update_status()
        prefix = "✅ " if ok else "⚠️ "
        self.status.set(prefix + msg.splitlines()[0])
        if not ok:
            messagebox.showwarning("مزامنة", msg)

    # ---------------------------------------------- إحصائيات / تقارير / PR --

    def export_yaml(self):
        out = self.repo / REPORTS_DIR / "annotation_stats.yaml"
        export_stats_yaml(self.lm.log, out)
        messagebox.showinfo("YAML", f"تم تصدير الإحصائيات إلى:\n{out}")

    def show_stats(self):
        stats = build_stats(self.lm.log)
        win = tk.Toplevel(self.root)
        win.title("إحصائيات الأنوتيشن")
        win.geometry("640x560")
        txt = tk.Text(win, font=("Courier", 10))
        txt.pack(fill=tk.BOTH, expand=True)
        lines = [
            f"إجمالي الصور المعالَجة : {stats['total_images_annotated']} / {len(self.images)}",
            f"إجمالي الصناديق        : {stats['total_boxes']}",
            "",
            "حسب المطوّر:",
        ]
        for dev, n in stats["images_per_developer"].items():
            lines.append(f"  {dev:<20} {n}")
        lines += ["", "الصناديق لكل كلاس (المجموع):"]
        for cls, n in sorted(stats["boxes_per_class_total"].items(), key=lambda kv: -kv[1]):
            bar = "█" * min(n, 40)
            lines.append(f"  {cls:<20} {n:>5}  {bar}")
        active = self.lm.log.get("locks", {})
        lines += ["", f"أقفال نشطة حالياً: {len(active)}"]
        for stem, lk in sorted(active.items()):
            lines.append(f"  {stem}  ← {lk.get('locked_by')} (حتى {lk.get('expires_at')})")
        txt.insert("1.0", "\n".join(lines))
        txt.configure(state="disabled")

    def report_pr(self):
        if not messagebox.askyesno(
                "تقرير + PR",
                "سيتم توليد تقرير Markdown + ملف YAML، إنشاء فرع جديد، رفعه إلى GitHub وفتح Pull Request.\n\nمتابعة؟"):
            return
        self.status.set("جارِ توليد التقرير ورفعه…")
        self.root.update_idletasks()

        def worker():
            stats_path = export_stats_yaml(
                self.lm.log, self.repo / REPORTS_DIR / "annotation_stats.yaml")
            md = build_markdown_report(self.lm.log, len(self.images))
            msg = publish_report_pr(self.lm, md, stats_path)
            self.root.after(0, lambda: (self._update_status(),
                                        messagebox.showinfo("تقرير + PR", msg)))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------- إغلاق --

    def on_close(self):
        if not self._confirm_leave():
            return
        # تحرير كل أقفالي عند الخروج ومزامنتها حتى لا تبقى صور محجوزة بلا داعٍ
        self.lm.release_all_mine()
        try:
            self.lm.sync(f"chore(locks): release locks on exit ({self.user})")
        except Exception:
            write_log(self.lm.log_path, self.lm.log)
        self.root.destroy()


# ----------------------------------------------------------------------------
# نقطة الدخول
# ----------------------------------------------------------------------------

def resolve_user(repo: Path) -> str:
    """اسم المطوّر: من الإعداد المحفوظ، أو git user.name، أو اسم النظام."""
    if USER_CFG.exists():
        try:
            name = json.loads(USER_CFG.read_text(encoding="utf-8")).get("name", "").strip()
            if name:
                return name
        except (json.JSONDecodeError, OSError):
            pass
    gitname = subprocess.run(["git", "config", "user.name"], cwd=repo,
                             capture_output=True, text=True).stdout.strip()
    return gitname or getpass.getuser()


def ask_user_name(root: tk.Tk, default: str) -> str:
    from tkinter import simpledialog
    name = simpledialog.askstring(
        "اسم المطوّر", "أدخل اسمك (سيظهر في السجل والأقفال):",
        initialvalue=default, parent=root)
    name = (name or default).strip() or default
    try:
        USER_CFG.write_text(json.dumps({"name": name}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return name


def main():
    ap = argparse.ArgumentParser(description="WDA Team Annotation Tool")
    ap.add_argument("--repo", type=Path, default=Path.cwd(),
                    help="مسار جذر مستودع War-Damage-Assessment (افتراضياً: المجلد الحالي)")
    ap.add_argument("--user", type=str, default=None, help="اسم المطوّر (اختياري)")
    args = ap.parse_args()

    repo = args.repo.resolve()
    if not (repo / "data").exists():
        print(f"تحذير: لم يُعثر على مجلد data داخل {repo} — تأكد أنك في جذر الريبو "
              f"وأنك نفّذت `dvc pull` لسحب الصور.", file=sys.stderr)

    root = tk.Tk()
    user = args.user or ask_user_name(root, resolve_user(repo))
    app = AnnotationApp(root, repo, user)
    root.mainloop()


if __name__ == "__main__":
    main()

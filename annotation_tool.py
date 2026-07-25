#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
War-Damage-Assessment — Team Annotation Tool
=============================================
أداة أنوتيشن جماعية لمشروع تقييم أضرار المباني.

- ترسم Bounding Boxes عادية (YOLO) ومدوّرة (YOLO-OBB).
- تدير الأقفال (Locks) بين أعضاء الفريق عبر processed_log.json المتزامن مع GitHub.
- تحفظ الـ labels في data/annotations (مُدارة عبر git مباشرة — ملفات نصية صغيرة)
  وتُرفع تلقائياً مع كل "حفظ + مزامنة". الصور الخام فقط هي المُدارة عبر DVC.
- Idempotency عبر MD5 + سجل الصور المعالجة.
- تصدير إحصائيات YAML وتقرير Markdown مع فتح Pull Request.

الاستخدام (من داخل البيئة الافتراضية للمشروع):
    .venv\\Scripts\\activate           # Windows  (أو: source .venv/bin/activate)
    python annotation_tool.py          # من جذر الريبو
    python annotation_tool.py --repo /path/to/War-Damage-Assessment

المتطلبات:
    pip install -r requirements.txt   (تتضمن dvc[gdrive] 3.x و Pillow و PyYAML)
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
        dmg = int(b.get("dmg", 0))
        x0, y0, x1, y1 = enclosing_rect(b)
        cx, cy = nx((x0 + x1) / 2), ny((y0 + y1) / 2)
        w, h = nx(x1 - x0), ny(y1 - y0)
        std_lines.append(f"{b['cls']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {dmg}")

        pts = box_corners(b)
        flat = " ".join(f"{nx(px):.6f} {ny(py):.6f}" for px, py in pts)
        obb_lines.append(f"{b['cls']} {flat} {dmg}")

    (repo / LABELS_DIR / f"{stem}.txt").write_text("\n".join(std_lines) + ("\n" if std_lines else ""), encoding="utf-8")
    (repo / LABELS_OBB_DIR / f"{stem}.txt").write_text("\n".join(obb_lines) + ("\n" if obb_lines else ""), encoding="utf-8")


def load_labels(stem: str, img_w: int, img_h: int, repo: Path) -> list[dict]:
    obb = repo / LABELS_OBB_DIR / f"{stem}.txt"
    std = repo / LABELS_DIR / f"{stem}.txt"
    boxes: list[dict] = []
    if obb.exists():
        for line in obb.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) not in (9, 10):
                continue
            cls = int(float(parts[0]))
            dmg = int(float(parts[9])) if len(parts) == 10 else 0
            vals = list(map(float, parts[1:9]))
            pts = [(vals[i] * img_w, vals[i + 1] * img_h) for i in range(0, 8, 2)]
            b = corners_to_box(pts, cls)
            if abs(b["angle"]) < 0.05:
                b["angle"] = 0.0
            b["dmg"] = dmg
            boxes.append(b)
    elif std.exists():
        for line in std.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) not in (5, 6):
                continue
            cls = int(float(parts[0]))
            dmg = int(float(parts[5])) if len(parts) == 6 else 0
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append({"cls": cls, "cx": cx * img_w, "cy": cy * img_h,
                          "w": w * img_w, "h": h * img_h, "angle": 0.0, "dmg": dmg})
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

    def refresh_from_remote(self) -> None:
        """جلب أحدث سجل من origin ودمجه محلياً — تُستدعى قبل الإحصائيات والتقارير
        حتى تعكس آخر عمل الفريق كله وليس النسخة المحلية فقط."""
        if not self.git_available():
            return
        self.log = merge_logs(self.log, self._remote_log())
        write_log(self.log_path, self.log)

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

            # ملفات labels تُدار في git مباشرة. نضيفها بـ -f احترازاً: لو بقي من
            # تصميم سابق سطر يتجاهل data/annotations في .gitignore (أو مؤشّر DVC
            # قديم) لكان git add العادي يتجاهلها صامتاً فلا تصل GitHub.
            annot = ANNOT_DIR.as_posix()
            forced = [p for p in (extra_paths or [])
                      if p.as_posix() == annot or p.as_posix().startswith(annot + "/")]
            normal = [str(LOG_FILE)] + [str(p) for p in (extra_paths or []) if p not in forced]
            self._git("add", *normal)
            if forced:
                self._git("add", "-f", *[str(p) for p in forced])
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
            "damaged_boxes": sum(int(b.get("dmg", 0)) for b in boxes),
            "annotated_by": self.user,
            "class_boxes": class_boxes,
        }

    def remove_image_record(self, stem: str) -> None:
        """إزالة كل أثر لصورة محذوفة من السجل: من annotated و augmented والأقفال."""
        for section in ("annotated", "augmented", "locks"):
            self.log.get(section, {}).pop(stem, None)

# ----------------------------------------------------------------------------
# استعادة الإحصائيات من ملفات الـ labels
#   إن فُقد قسم annotated من processed_log.json (خلل/حذف) بينما ملفات الـ labels
#   سليمة، نعيد بناء السجل بقراءة كل ملف label وعدّ صناديقه وكلاساته وحالة الدمار.
#   يتطلب معرفة أبعاد كل صورة (لأن الإحداثيات منسّقة 0–1) — نقرأها من الصور نفسها.
# ----------------------------------------------------------------------------

def _count_label_file(path: Path) -> tuple[int, int, dict, bool] | None:
    """يقرأ ملف label (std أو obb) ويرجع (عدد الصناديق, عدد المدمّرة,
    عدّ لكل كلاس, هل obb). لا يحتاج أبعاد الصورة — العدّ لا يعتمد عليها.
    يرجع None إذا كان الملف فارغاً أو غير صالح."""
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.split()]
    except OSError:
        return None
    class_boxes = {name: 0 for name in CLASSES}
    n_boxes = 0
    n_damaged = 0
    is_obb = False
    for line in lines:
        parts = line.split()
        # std: 5 (قديم) أو 6 (مع dmg) | obb: 9 (قديم) أو 10 (مع dmg)
        if len(parts) in (9, 10):
            is_obb = True
            dmg_idx = 9
        elif len(parts) in (5, 6):
            dmg_idx = 5
        else:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        dmg = 0
        if len(parts) == dmg_idx + 1:
            try:
                dmg = int(float(parts[dmg_idx]))
            except ValueError:
                dmg = 0
        n_boxes += 1
        n_damaged += 1 if dmg else 0
        if 0 <= cls < len(CLASSES):
            class_boxes[CLASSES[cls]] += 1
    return n_boxes, n_damaged, class_boxes, is_obb


def scan_labels_for_recovery(repo: Path) -> dict[str, dict]:
    """يمسح مجلدي labels/ و labels_obb/ ويبني قاموس {stem: entry} جاهز
    للدمج في log['annotated']. يفضّل نسخة OBB (الأكمل) عند وجود الاثنين.
    stem = اسم الصورة بلا امتداد (يطابق اسم ملف الـ label)."""
    result: dict[str, dict] = {}
    # نبدأ بـ OBB (المصدر الكامل) ثم نكمّل بأي std لا يوجد له obb
    for folder, prefer in ((repo / LABELS_OBB_DIR, True), (repo / LABELS_DIR, False)):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.txt")):
            stem = path.stem
            if stem in result and not prefer:
                continue  # لدينا نسخة OBB أفضل بالفعل
            counted = _count_label_file(path)
            if counted is None:
                continue
            n_boxes, n_damaged, class_boxes, is_obb = counted
            result[stem] = {
                "boxes": n_boxes,
                "damaged_boxes": n_damaged,
                "class_boxes": class_boxes,
                "_source": "obb" if is_obb else "std",
            }
    return result


def find_image_for_stem(repo: Path, stem: str, image_index: dict[str, Path] | None = None) -> Path | None:
    """يعثر على ملف الصورة المقابل لـ stem داخل data/raw (لأي امتداد مدعوم)."""
    if image_index is not None:
        return image_index.get(stem)
    raw = repo / RAW_DIR
    if not raw.exists():
        return None
    for ext in IMG_EXTS:
        cand = raw / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def recover_annotated_log(repo: Path, log: dict, default_dev: str,
                          compute_hash: bool = True,
                          progress=None) -> tuple[int, int, list[str]]:
    """يعيد بناء إدخالات annotated المفقودة من ملفات الـ labels.

    - لا يلمس الإدخالات الموجودة سلفاً (لا نكسر ما هو سليم) إلا لإكمال حقول ناقصة.
    - لكل label بلا إدخال في annotated: ننشئ إدخالاً كاملاً. نحسب MD5 للصورة
      المقابلة إن وُجدت (للحفاظ على Idempotency)، وإلا نترك hash فارغاً.
    - annotated_by للإدخالات المستعادة = default_dev (لا نعرف الأصل الحقيقي).

    يرجع (عدد المستعاد, عدد المتخطّى الموجود سلفاً, قائمة stems بلا صورة مطابقة).
    """
    recovered = scan_labels_for_recovery(repo)
    raw = repo / RAW_DIR
    image_index = {p.stem: p for p in raw.iterdir()
                   if p.suffix.lower() in IMG_EXTS} if raw.exists() else {}

    annotated = log.setdefault("annotated", {})
    added = 0
    skipped = 0
    missing_images: list[str] = []
    items = list(recovered.items())
    for i, (stem, info) in enumerate(items):
        if progress and i % 25 == 0:
            progress(i, len(items))
        img_path = image_index.get(stem)
        if img_path is None:
            missing_images.append(stem)

        existing = annotated.get(stem)
        if existing is not None:
            # إدخال موجود — نكمّل الحقول الناقصة فقط دون الكتابة فوق الموجود
            existing.setdefault("boxes", info["boxes"])
            existing.setdefault("damaged_boxes", info["damaged_boxes"])
            existing.setdefault("class_boxes", info["class_boxes"])
            existing.setdefault("annotated_by", default_dev)
            existing.setdefault("timestamp", iso(utcnow()))
            if compute_hash and not existing.get("hash") and img_path is not None:
                try:
                    existing["hash"] = file_md5(img_path)
                except OSError:
                    pass
            skipped += 1
            continue

        entry = {
            "hash": "",
            "timestamp": iso(utcnow()),
            "boxes": info["boxes"],
            "damaged_boxes": info["damaged_boxes"],
            "annotated_by": default_dev,
            "class_boxes": info["class_boxes"],
            "recovered": True,   # وسم أن هذا الإدخال أُعيد بناؤه من ملفات الـ labels
        }
        if compute_hash and img_path is not None:
            try:
                entry["hash"] = file_md5(img_path)
            except OSError:
                pass
        annotated[stem] = entry
        added += 1
    if progress:
        progress(len(items), len(items))
    return added, skipped, missing_images


# ----------------------------------------------------------------------------
# الإحصائيات والتقارير
# ----------------------------------------------------------------------------

def build_stats(log: dict) -> dict:
    annotated = log.get("annotated", {})
    totals = {name: 0 for name in CLASSES}
    per_dev: dict[str, int] = {}
    total_boxes = 0
    total_damaged = 0
    for entry in annotated.values():
        total_boxes += entry.get("boxes", 0)
        total_damaged += entry.get("damaged_boxes", 0)
        dev = entry.get("annotated_by", "unknown")
        per_dev[dev] = per_dev.get(dev, 0) + 1
        for cls, n in entry.get("class_boxes", {}).items():
            if cls in totals:
                totals[cls] += n
    return {
        "generated_at": iso(utcnow()),
        "total_images_annotated": len(annotated),
        "total_boxes": total_boxes,
        "total_damaged_boxes": total_damaged,
        "total_intact_boxes": total_boxes - total_damaged,
        "images_per_developer": dict(sorted(per_dev.items(), key=lambda kv: -kv[1])),
        "boxes_per_class_total": totals,
        "per_image": {
            stem: {
                "annotated_by": e.get("annotated_by"),
                "timestamp": e.get("timestamp"),
                "boxes": e.get("boxes", 0),
                "damaged_boxes": e.get("damaged_boxes", 0),
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
        f"- **Total bounding boxes:** {stats['total_boxes']} "
        f"(damaged: {stats['total_damaged_boxes']}, intact: {stats['total_intact_boxes']})",
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
# التكامل مع DVC
#   - الصور الخام (data/raw) فقط هي المُدارة عبر DVC — git يحمل مؤشّر raw.dvc.
#   - الـ labels (data/annotations) ملفات نصية صغيرة تُدار في git مباشرة.
#   - سير رفع صور جديدة: dvc add → dvc push (البيانات) → git commit/push (المؤشّر).
#     dvc push قبل git حتى لا يصل المؤشّر للفريق قبل البيانات نفسها.
# ----------------------------------------------------------------------------

class DvcManager:
    # قفل على مستوى الصنف (وليس النسخة) يضمن ألّا يعمل أمرا dvc معاً أبداً داخل
    # التطبيق مهما تعددت النسخ أو الـ threads. هذا هو سبب خطأ "Unable to acquire
    # lock": التطبيق يشغّل dvc status/pull من threads خلفية فتتصادم على قفل DVC
    # الداخلي (.dvc/tmp/rwlock)، بينما من سطر الأوامر يُنفَّذ أمر واحد فقط كل مرة.
    _cmd_lock = threading.Lock()

    def __init__(self, repo: Path):
        self.repo = repo
        self.dvc_cmd = self._resolve_dvc()

    def _resolve_dvc(self) -> str:
        """يفضّل dvc الخاص ببيئة المشروع الافتراضية (.venv) حتى لا تُستخدم
        نسخة نظام قديمة غير متوافقة مع صيغة ملفات .dvc الحالية (DVC 3)."""
        exe = "dvc.exe" if os.name == "nt" else "dvc"
        candidates = [
            Path(sys.executable).parent / exe,       # بيئة تشغيل الأداة نفسها
            self.repo / ".venv" / "Scripts" / exe,   # venv المشروع (Windows)
            self.repo / ".venv" / "bin" / "dvc",     # venv المشروع (Linux/macOS)
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return "dvc"

    def _clear_stale_lock(self) -> bool:
        """يحذف ملفات قفل DVC المتبقية من عملية سابقة انتهت فجأة (إغلاق التطبيق
        أثناء pull، أو thread قُتِل). آمن لأننا نستدعيه فقط ونحن ممسكون بـ
        _cmd_lock — أي لا توجد عملية dvc أطلقناها نحن تعمل الآن. يُعالِج الشِق
        الثاني من رسالة الخطأ: '...or was terminated abruptly'."""
        tmp = self.repo / ".dvc" / "tmp"
        if not tmp.exists():
            return False
        removed = False
        for name in ("lock", "rwlock", "rwlock.lock"):
            p = tmp / name
            try:
                if p.exists():
                    p.unlink()
                    removed = True
            except OSError:
                pass  # قد يكون مقفولاً فعلاً من عملية حيّة خارج التطبيق — نتركه
        return removed

    def _dvc(self, *args: str, timeout: int | None = None,
             retry_on_lock: bool = True) -> subprocess.CompletedProcess:
        """تشغيل أمر dvc مع تسلسل صارم (قفل واحد) + تنظيف الأقفال العالقة وإعادة
        محاولة واحدة عند خطأ القفل. كل أوامر dvc تمرّ من هنا."""
        with DvcManager._cmd_lock:
            r = subprocess.run([self.dvc_cmd, *args], cwd=self.repo,
                               capture_output=True, text=True, timeout=timeout)
            if retry_on_lock and r.returncode != 0:
                out = (r.stdout + r.stderr).lower()
                if "unable to acquire lock" in out or "another dvc process" in out:
                    # لا عملية dvc لنا تعمل الآن (نحن داخل القفل) → القفل عالق: ننظّفه ونعيد مرة
                    self._clear_stale_lock()
                    r = subprocess.run([self.dvc_cmd, *args], cwd=self.repo,
                                       capture_output=True, text=True, timeout=timeout)
            return r

    def available(self) -> bool:
        try:
            return self._dvc("--version", timeout=30).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def raw_pointer_nfiles(self) -> int | None:
        """عدد الملفات المسجّل في مؤشّر raw.dvc (nfiles)، أو None إن تعذّر."""
        ptr = self.repo / "data" / "raw.dvc"
        if not ptr.exists():
            return None
        try:
            for line in ptr.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("nfiles:"):
                    return int(s.split(":", 1)[1].strip())
        except (OSError, ValueError):
            pass
        return None

    def raw_disk_image_count(self) -> int:
        raw = self.repo / RAW_DIR
        if not raw.exists():
            return 0
        return sum(1 for p in raw.iterdir() if p.suffix.lower() in IMG_EXTS)

    def raw_changed(self) -> bool:
        """هل يختلف محتوى data/raw عمّا هو مسجّل في المؤشّر (dvc status)؟"""
        ptr = self.repo / "data" / "raw.dvc"
        if not ptr.exists():
            return (self.repo / RAW_DIR).exists()
        r = self._dvc("status", str(Path("data") / "raw.dvc"))
        out = (r.stdout + r.stderr).lower()
        return r.returncode == 0 and "up to date" not in out and out.strip() != ""

    def raw_state(self) -> str:
        """تصنيف حالة data/raw مقابل المؤشّر — يحدّد الإجراء الآمن:
          'clean'    : مطابق للمؤشّر — لا شيء.
          'has_new'  : على القرص صور أكثر من المؤشّر (إضافات محلية) → يجب الرفع،
                       وممنوع dvc pull لأنه سيحذفها.
          'missing'  : على القرص صور أقل (ناقصة) → السحب آمن ومطلوب.
          'unknown'  : تغيّر غامض بنفس العدد → نعامله كإضافة (الأحوط: لا حذف).
        """
        if not self.raw_changed():
            return "clean"
        nf = self.raw_pointer_nfiles()
        dc = self.raw_disk_image_count()
        if nf is None:
            return "unknown"
        if dc > nf:
            return "has_new"
        if dc < nf:
            return "missing"
        return "unknown"

    def pull_raw(self, state: str | None = None) -> tuple[bool, str]:
        """سحب صور المؤشّر. أمان: نرفض السحب إن وُجدت إضافات محلية غير مرفوعة
        لأن dvc pull ينفّذ checkout يحذف أي ملف ليس في المؤشّر.
        state: حالة raw محسوبة مسبقاً (لتفادي استدعاء dvc status مكرر)."""
        if state is None:
            state = self.raw_state()
        if state in ("has_new", "unknown"):
            return False, ("توجد صور محلية جديدة في data/raw لم تُرفع بعد. "
                           "أُلغي dvc pull حتى لا تُحذف — ارفعها أولاً بزر "
                           "«☁ مزامنة الصور».")
        r = self._dvc("pull", str(Path("data") / "raw.dvc"))
        if r.returncode == 0:
            return True, "تم سحب data/raw من DVC remote بنجاح."
        return False, f"فشل dvc pull:\n{(r.stderr or r.stdout).strip()[-400:]}"

    def sync_raw(self, lm: "LockManager") -> tuple[bool, str]:
        """رفع الصور الخام الجديدة: dvc add → dvc push → git (المؤشّر)."""
        add = self._dvc("add", str(RAW_DIR))
        if add.returncode != 0:
            return False, f"فشل dvc add data/raw:\n{(add.stderr or add.stdout).strip()[-400:]}"

        push = self._dvc("push", str(Path("data") / "raw.dvc"))
        if push.returncode != 0:
            return False, ("فشل dvc push — تحقق من اعتمادات Google Drive في "
                           ".dvc/config.local أو ثبّت:  pip install \"dvc[gdrive]\"\n"
                           f"{(push.stderr or push.stdout).strip()[-400:]}")

        extra = [Path("data") / "raw.dvc"]
        if (self.repo / "data" / ".gitignore").exists():
            extra.append(Path("data") / ".gitignore")
        ok, git_msg = lm.sync(f"data: add raw images via DVC ({lm.user})",
                              extra_paths=extra)
        if not ok:
            return False, f"تم dvc add/push لكن فشلت مزامنة git:\n{git_msg}"
        if self.raw_changed():
            return False, ("رُفعت صورك لكن مؤشّر raw.dvc تغيّر أثناء المزامنة "
                           "(زميل رفع صوراً في نفس اللحظة) — أعد المزامنة بعد قليل.")
        return True, "تم رفع الصور الجديدة: dvc add ✓  dvc push ✓  git push ✓"

    def delete_image_from_dvc(self, image_path: Path) -> tuple[bool, str]:
        """حذف صورة نهائياً من DVC: حذف الملف → dvc add (يحدّث المؤشّر ليُسقطها) →
        dvc push (يرفع المانيفست الجديد). لا يلمس git — المُنادي يتكفّل بمزامنته.

        ملاحظة أمان: هذا الحذف قابل للاسترجاع — blob الصورة يبقى في الـ remote
        cache، فيمكن التراجع لاحقاً بـ git revert للمؤشّر ثم dvc checkout، ما لم
        يُشغّل أحدهم dvc gc على الـ cloud."""
        if not image_path.exists():
            return False, f"ملف الصورة غير موجود: {image_path.name}"
        try:
            image_path.unlink()
        except OSError as e:
            # على ويندوز قد يفشل الحذف إن كان الملف مفتوحاً في مكان آخر
            return False, (f"تعذّر حذف ملف الصورة (قد يكون مفتوحاً في برنامج آخر): "
                           f"{e}")
        add = self._dvc("add", str(RAW_DIR))
        if add.returncode != 0:
            return False, (f"حُذف الملف محلياً لكن فشل dvc add لتحديث المؤشّر:\n"
                           f"{(add.stderr or add.stdout).strip()[-400:]}")
        push = self._dvc("push", str(Path("data") / "raw.dvc"))
        if push.returncode != 0:
            return False, (f"حُذف الملف وحُدّث المؤشّر محلياً لكن فشل dvc push:\n"
                           f"{(push.stderr or push.stdout).strip()[-400:]}")
        return True, "تم حذف الصورة من DVC (dvc add ✓  dvc push ✓)"




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
        self.dvc = DvcManager(repo)
        self.dvc_busy = False     # عملية dvc جارية (تمنع تداخل الأوامر من الواجهة)

        self.images: list[Path] = []
        self.idx = -1
        self.img: Image.Image | None = None
        self.tk_img = None
        self.img_w = self.img_h = 0

        self.boxes: list[dict] = []
        self.selected: int | None = None
        self.clipboard: list[dict] = []
        self.dirty = False
        self.view_only = False    # الصورة الحالية للعرض فقط (محجوزة لزميل)
        self.current_class = 0
        self.current_dmg = 0      # الحالة الافتراضية للصناديق الجديدة: 0 سليم / 1 مدمّر

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
        has_images = raw.exists() and any(
            p.suffix.lower() in IMG_EXTS for p in raw.iterdir())
        if has_images:
            self.load_folder(raw)
        if self.dvc.available():
            # دائماً: سحب أي تحديثات للصور في الخلفية (no-op إن لم يتغير شيء)
            self._pull_raw_updates(raw)
        elif not has_images:
            messagebox.showwarning(
                "DVC", "مجلد data/raw فارغ وأمر dvc غير متوفر.\n"
                       "ثبّته ثم أعد التشغيل:  pip install \"dvc[gdrive]\"")

    def _pull_raw_updates(self, raw: Path):
        """في الخلفية: سحب تحديثات الصور من الزملاء تلقائياً — بأمان.
        إن وُجدت إضافات محلية غير مرفوعة لا نسحب (لأن السحب يحذفها) بل ننبّه فقط."""
        state = self.dvc.raw_state()
        if state in ("has_new", "unknown"):
            # لدى المستخدم صور جديدة لم تُرفع — لا نلمس القرص، فقط ننبّه
            self.status.set("⬆ لديك صور جديدة في data/raw لم تُرفع — اضغط «☁ مزامنة الصور» لرفعها")
            return
        self.status.set("جارِ التحقق من تحديثات الصور (dvc pull)…")
        self.dvc_busy = True

        def worker():
            ok, msg = self.dvc.pull_raw(state=state)

            def done():
                self.dvc_busy = False
                if not ok:
                    self.status.set("⚠️ " + msg.splitlines()[0])
                    messagebox.showwarning(
                        "DVC", msg + "\n\nتحقق من تثبيت dvc واعتمادات Google Drive "
                                     "في .dvc/config.local")
                    return
                on_disk = sorted(p for p in raw.iterdir()
                                 if p.suffix.lower() in IMG_EXTS) if raw.exists() else []
                if len(on_disk) != len(self.images):
                    if self.idx < 0:
                        self.load_folder(raw)
                        self.status.set(f"✅ وصلت صور جديدة — الإجمالي الآن {len(self.images)}")
                    else:
                        # لا نقاطع المستخدم أثناء عمله — نكتفي بالتنبيه
                        self.status.set(f"✅ وصلت صور جديدة ({len(on_disk)}) — "
                                        "أعد فتح المجلد (📁) لرؤيتها")
                elif self.idx >= 0:
                    self._update_status()
                else:
                    self.status.set("✅ الصور محدّثة — اختر مجلد الصور للبدء…")
            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

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

        # الانتقال لترتيب محدد: خانة إدخال رقم الصورة (1-based) + زر اذهب
        tk.Label(top, text="اذهب لرقم:").pack(side=tk.LEFT, padx=(8, 1))
        self.goto_var = tk.StringVar()
        goto_entry = tk.Entry(top, textvariable=self.goto_var, width=6)
        goto_entry.pack(side=tk.LEFT)
        goto_entry.bind("<Return>", lambda e: self.goto_index_entry())
        tk.Button(top, text="اذهب", command=self.goto_index_entry).pack(side=tk.LEFT, padx=1)

        # علامة البداية الشخصية: كل مطوّر يحدد من أين يبدأ
        tk.Button(top, text="⭐ ضع بدايتي هنا", command=self.set_start_marker).pack(side=tk.LEFT, padx=(8, 1))
        tk.Button(top, text="↩ اذهب لبدايتي", command=self.goto_start_marker).pack(side=tk.LEFT, padx=1)

        tk.Button(top, text="💾 حفظ + مزامنة (Ctrl+S)", bg="#d1f0d1",
                  command=self.save_current).pack(side=tk.LEFT, padx=8)
        tk.Button(top, text="🗑 حذف المحدد (Del)", command=self.delete_selected).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="🧹 مسح الكل", command=self.clear_all).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="❌ حذف الصورة نهائياً", bg="#f5b7b1",
                  command=self.delete_current_image).pack(side=tk.LEFT, padx=8)
        tk.Button(top, text="📊 الإحصائيات", command=self.show_stats).pack(side=tk.RIGHT, padx=2)
        tk.Button(top, text="📄 تقرير + PR", command=self.report_pr).pack(side=tk.RIGHT, padx=2)
        tk.Button(top, text="🟡 YAML", command=self.export_yaml).pack(side=tk.RIGHT, padx=2)
        tk.Button(top, text="♻ استعادة الإحصائيات", bg="#ffe0b3",
                  command=self.recover_stats).pack(side=tk.RIGHT, padx=2)
        self.dvc_btn = tk.Button(top, text="☁ مزامنة الصور (DVC)", bg="#d6e4ff",
                                 command=self.sync_dvc)
        self.dvc_btn.pack(side=tk.RIGHT, padx=8)

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

        self.dmg_var = tk.IntVar(value=0)
        self.dmg_chk = tk.Checkbutton(
            side, text="💥 مدمّر (X)", variable=self.dmg_var,
            anchor="w", command=self.on_dmg_toggle)
        self.dmg_chk.pack(fill=tk.X, pady=(2, 0))

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
        r.bind("x", lambda e: self.toggle_dmg_key())
        r.bind("X", lambda e: self.toggle_dmg_key())
        r.bind("<Control-c>", lambda e: self.copy_selected())
        r.bind("<Control-v>", lambda e: self.paste_clipboard())
        r.bind("<Delete>", lambda e: self.delete_selected())
        r.bind("<Escape>", lambda e: self.select_box(None))
        r.bind("a", lambda e: self.prev_image())
        r.bind("d", lambda e: self.next_image())
        r.bind("<Left>", lambda e: self.prev_image())
        r.bind("<Right>", lambda e: self.next_image())
        r.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_dmg_toggle(self):
        """Checkbox أو مفتاح X: يبدّل حالة الـ box المحدد، أو الافتراضي للجديد."""
        val = self.dmg_var.get()
        if self.selected is not None:
            self.boxes[self.selected]["dmg"] = val
            self.dirty = True
            self._refresh_box_list()
            self.redraw()
        else:
            self.current_dmg = val
        self._update_status()

    def toggle_dmg_key(self):
        self.dmg_var.set(0 if self.dmg_var.get() else 1)
        self.on_dmg_toggle()

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

    def load_folder(self, folder: Path, honor_marker: bool = True):
        self.images = sorted(p for p in folder.iterdir()
                             if p.suffix.lower() in IMG_EXTS)
        if not self.images:
            messagebox.showwarning("لا صور", f"لا توجد صور في:\n{folder}")
            return
        self.idx = -1
        # إن كان للمطوّر علامة بداية محفوظة نفتحها، وإلا أول صورة غير معالَجة
        if honor_marker and self.goto_start_marker(silent=True):
            return
        self.next_unannotated(from_start=True)

    def _stem(self, i: int | None = None) -> str:
        i = self.idx if i is None else i
        return self.images[i].stem

    def _confirm_leave(self) -> bool:
        if not self.dirty or self.view_only:
            return True   # في وضع العرض فقط لا حفظ ولا تحذير
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
        self.view_only = False    # وضع القراءة فقط (صورة محجوزة لزميل نتصفّحها دون حجز)
        if owner is not None and owner != self.user and not force:
            # بدل الرفض التام: نتيح فتحها للعرض فقط (دون حجز ودون إمكانية حفظ)
            # حتى يمكن الانتقال لأي رقم وتفقّد أي صورة، مع صون ضمان القفل الجماعي.
            if not messagebox.askyesno(
                    "صورة محجوزة",
                    f"الصورة {stem} محجوزة حالياً بواسطة: {owner}\n"
                    f"(ينتهي القفل تلقائياً بعد ساعتين من حجزه)\n\n"
                    f"فتحها للعرض فقط (بدون تعديل أو حفظ)؟"):
                return
            self.view_only = True
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

        # في وضع العرض فقط لا نحجز شيئاً (الصورة محجوزة لزميل) ولا نلمس أقفاله
        if not self.view_only:
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

    def goto_index_entry(self):
        """الانتقال للصورة برقم ترتيبها (1-based) من خانة الإدخال."""
        if not self.images:
            messagebox.showwarning("لا صور", "افتح مجلد صور أولاً.")
            return
        raw = self.goto_var.get().strip()
        if not raw:
            return
        try:
            n = int(raw)
        except ValueError:
            messagebox.showwarning("رقم غير صالح", f"أدخل رقماً بين 1 و {len(self.images)}.")
            return
        if not (1 <= n <= len(self.images)):
            messagebox.showwarning("خارج النطاق",
                                   f"الرقم يجب أن يكون بين 1 و {len(self.images)}.")
            return
        self.goto_var.set("")
        self.open_index(n - 1)   # التحويل من 1-based إلى فهرس 0-based

    # ------------------------------------------------------ علامة البداية --

    def _start_marker_key(self) -> str:
        """مفتاح يربط علامة البداية بهذا الريبو تحديداً (كي لا تختلط المشاريع)."""
        return str(self.repo.resolve())

    def _load_start_markers(self) -> dict:
        if not USER_CFG.exists():
            return {}
        try:
            data = json.loads(USER_CFG.read_text(encoding="utf-8"))
            return data.get("start_markers", {})
        except (json.JSONDecodeError, OSError):
            return {}

    def set_start_marker(self):
        """يحفظ الصورة الحالية كنقطة بداية شخصية لهذا المطوّر (تبقى بين الجلسات)."""
        if self.idx < 0:
            messagebox.showwarning("لا صورة", "افتح صورة أولاً لتحديدها كبداية.")
            return
        stem = self._stem()
        try:
            data = json.loads(USER_CFG.read_text(encoding="utf-8")) if USER_CFG.exists() else {}
        except (json.JSONDecodeError, OSError):
            data = {}
        markers = data.setdefault("start_markers", {})
        markers[self._start_marker_key()] = stem
        try:
            USER_CFG.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:
            messagebox.showwarning("خطأ", f"تعذّر حفظ العلامة:\n{e}")
            return
        self.status.set(f"⭐ حُفظت بدايتك عند الصورة [{self.idx + 1}] {stem} — "
                        "ستُفتح تلقائياً في تشغيلك القادم")

    def goto_start_marker(self, silent: bool = False) -> bool:
        """الانتقال لعلامة البداية المحفوظة لهذا المطوّر إن وُجدت وطابقت صورة."""
        markers = self._load_start_markers()
        stem = markers.get(self._start_marker_key())
        if not stem:
            if not silent:
                messagebox.showinfo("لا علامة",
                                    "لم تحدّد نقطة بداية بعد. افتح صورة واضغط «⭐ ضع بدايتي هنا».")
            return False
        for j, p in enumerate(self.images):
            if p.stem == stem:
                self.open_index(j)
                return True
        if not silent:
            messagebox.showwarning("غير موجودة",
                                   f"علامة البداية ({stem}) لا تطابق أي صورة في المجلد الحالي.")
        return False

    def _update_status(self):
        if self.idx < 0:
            return
        stem = self._stem()
        annotated = "✅ معالَجة" if self.lm.is_annotated(stem, self.images[self.idx]) else "⬜ غير معالَجة"
        nxt = self._find_next_free(self.idx + 1)
        reserved = self._stem(nxt) if nxt is not None else "—"
        vo = "  |  👁 عرض فقط" if self.view_only else ""
        self.status.set(
            f"[{self.idx + 1}/{len(self.images)}] {stem}  |  {annotated}{vo}  |  "
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
                               "w": 1.0, "h": 1.0, "angle": 0.0,
                               "dmg": self.current_dmg})
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
            self.dmg_var.set(int(self.boxes[i].get("dmg", 0)))
            self.box_list.selection_clear(0, tk.END)
            if i < self.box_list.size():
                self.box_list.selection_set(i)
        else:
            self.dmg_var.set(self.current_dmg)
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

    # ------------------------------------------------- حذف الصورة نهائياً --

    def delete_current_image(self):
        """حذف الصورة الحالية نهائياً: الملف من DVC + ملفات الـ label + سجلّها،
        ثم رفع كل ذلك إلى الفريق. عملية مدمّرة لكنها قابلة للاسترجاع بـ git revert."""
        if self.idx < 0 or self.img is None:
            messagebox.showwarning("لا صورة", "افتح صورة أولاً.")
            return
        if self.view_only:
            messagebox.showinfo("عرض فقط",
                                "هذه الصورة محجوزة لزميل (عرض فقط) — لا يمكنك حذفها.")
            return
        if not self.dvc.available():
            messagebox.showwarning(
                "DVC مطلوب",
                "الصور مُدارة عبر DVC، وحذفها نهائياً يتطلب توفّر أمر dvc.\n"
                "ثبّته ثم أعد المحاولة:  pip install \"dvc[gdrive]\"")
            return
        if self.dvc_busy:
            self.status.set("عملية DVC جارية — انتظر انتهاءها ثم أعد المحاولة.")
            return

        stem = self._stem()
        img_path = self.images[self.idx]

        # لا نحذف صورة يعمل عليها زميل الآن (قفل نشط لغيري)
        owner = self.lm.lock_owner(stem)
        if owner is not None and owner != self.user:
            messagebox.showwarning(
                "محجوزة لزميل",
                f"الصورة {stem} محجوزة حالياً بواسطة {owner} — لا تحذفها أثناء عمله عليها.\n"
                f"انتظر انتهاء قفله (ساعتان كحدّ أقصى) أو نسّق معه.")
            return

        n_labels = sum(1 for d in (LABELS_DIR, LABELS_OBB_DIR)
                       if (self.repo / d / f"{stem}.txt").exists())
        if not messagebox.askyesno(
                "⚠️ حذف نهائي للصورة",
                f"سيُحذف نهائياً وللجميع:\n\n"
                f"• الصورة: {img_path.name}\n"
                f"• {n_labels} ملف label مرتبط بها\n"
                f"• سجلّها في processed_log.json\n\n"
                f"يُحدَّث مؤشّر DVC ويُرفع إلى الفريق. عند مزامنتهم القادمة ستختفي الصورة من أجهزتهم.\n\n"
                f"↩ الحذف قابل للاسترجاع لاحقاً بـ git revert للمؤشّر ثم dvc checkout "
                f"(تبقى الصورة في مخزن DVC حتى يُشغَّل dvc gc).\n\n"
                f"متابعة الحذف؟"):
            return
        # تأكيد ثانٍ صريح — العملية تمسّ الجميع
        if not messagebox.askyesno("تأكيد أخير",
                                   f"تأكيد حذف «{img_path.name}» نهائياً من الريبو للجميع؟"):
            return

        self.dvc_busy = True
        # نُفلت مرجع الصورة قبل الحذف حتى لا يمنع ويندوز حذف ملف مفتوح
        self.img = None
        self.tk_img = None
        self.boxes = []
        self.selected = None
        self.dirty = False
        self.canvas.delete("all")
        self.status.set(f"جارِ حذف {img_path.name} نهائياً…")
        self.root.update_idletasks()

        def worker():
            # 1) أحدث سجل من GitHub أولاً (كي لا نطمس عمل زملاء)
            self.lm.refresh_from_remote()

            # 2) حذف ملفات الـ label (مُدارة في git)
            removed_labels = []
            for d in (LABELS_DIR, LABELS_OBB_DIR):
                p = self.repo / d / f"{stem}.txt"
                try:
                    if p.exists():
                        p.unlink()
                        removed_labels.append(str(d / f"{stem}.txt"))
                except OSError:
                    pass

            # 3) إزالة سجلّ الصورة من processed_log.json (annotated/locks)
            self.lm.remove_image_record(stem)
            write_log(self.lm.log_path, self.lm.log)

            # 4) حذف الصورة من DVC (unlink + dvc add + dvc push)
            ok_dvc, msg_dvc = self.dvc.delete_image_from_dvc(img_path)
            if not ok_dvc:
                # فشل DVC — نعيد السجل لأقرب حالة متسقة ونبلّغ
                self.root.after(0, lambda: self._after_delete(False, msg_dvc, stem, None))
                return

            # 5) مزامنة git: المؤشّر الجديد + حذف ملفات الـ label + السجل، في commit واحد
            extra = [Path("data") / "raw.dvc", ANNOT_DIR]
            if (self.repo / "data" / ".gitignore").exists():
                extra.append(Path("data") / ".gitignore")
            ok_git, msg_git = self.lm.sync(
                f"delete(image): remove {stem} from dataset ({self.user})",
                extra_paths=extra)
            combined = (msg_dvc + "\n" + msg_git) if not ok_git else msg_git
            self.root.after(0, lambda: self._after_delete(ok_git, combined, stem, img_path))

        threading.Thread(target=worker, daemon=True).start()

    def _after_delete(self, ok: bool, msg: str, stem: str, img_path):
        self.dvc_busy = False
        if not ok:
            self.status.set("⚠️ فشل الحذف — " + msg.splitlines()[0])
            messagebox.showwarning("حذف الصورة", msg)
            # نعيد فتح الصورة الحالية إن كانت ما زالت موجودة، وإلا ننتقل
            if img_path is not None and img_path.exists():
                self.open_index(self.idx)
            else:
                self._reopen_after_removal(stem)
            return
        # نجح: نزيل الصورة من القائمة وننتقل للتالية
        self.status.set("✅ حُذفت الصورة نهائياً — " + msg.splitlines()[-1])
        self._reopen_after_removal(stem)

    def _reopen_after_removal(self, stem: str):
        """يحدّث قائمة الصور بعد حذف واحدة وينتقل لصورة مناسبة."""
        old_idx = self.idx
        self.images = [p for p in self.images if p.stem != stem]
        if not self.images:
            self.idx = -1
            self.img = None
            self.tk_img = None
            self.boxes = []
            self.canvas.delete("all")
            self.status.set("لا صور متبقية في المجلد.")
            return
        # نفتح الصورة التي أخذت مكان المحذوفة (أو آخر صورة إن حذفنا الأخيرة)
        self.idx = -1   # نُجبر open_index على الفتح دون منطق "المغادرة"
        target = min(old_idx, len(self.images) - 1)
        self.open_index(target)

    def _refresh_box_list(self):
        self.box_list.delete(0, tk.END)
        for b in self.boxes:
            rot = f" ∠{b['angle']:.0f}°" if abs(b["angle"]) > 0.05 else ""
            state = "💥" if b.get("dmg", 0) else "✓"
            self.box_list.insert(tk.END, f"{state} {CLASSES[b['cls']]}{rot}")

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
            damaged = bool(b.get("dmg", 0))
            pts = [self.to_canvas(x, y) for x, y in box_corners(b)]
            flat = [v for p in pts for v in p]
            width = 3 if i == self.selected else 2
            dash = (6, 4) if damaged else None
            c.create_polygon(*flat, outline=color, fill="", width=width, dash=dash)
            lx, ly = pts[0]
            label = ("💥 " if damaged else "") + CLASSES[b["cls"]]
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
        if self.view_only:
            messagebox.showinfo(
                "عرض فقط",
                "هذه الصورة محجوزة لزميل وأنت تتصفّحها للعرض فقط — لا يمكن الحفظ.\n"
                "انتقل لصورة أخرى للعمل عليها.")
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
            # ANNOT_DIR ضمن الرفع: كل حفظ يرفع ملفات الـ labels مع السجل في commit واحد
            ok, msg = self.lm.sync(
                f"annotate({self.user}): {stem} — {len(self.boxes)} boxes",
                extra_paths=[ANNOT_DIR])
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
        self.status.set("جارِ تحديث السجل من GitHub ثم التصدير…")
        self.root.update_idletasks()

        def worker():
            self.lm.refresh_from_remote()
            out = self.repo / REPORTS_DIR / "annotation_stats.yaml"
            export_stats_yaml(self.lm.log, out)
            self.root.after(0, lambda: (
                self._update_status(),
                messagebox.showinfo("YAML", f"تم تصدير الإحصائيات إلى:\n{out}")))

        threading.Thread(target=worker, daemon=True).start()

    def recover_stats(self):
        """استعادة قسم annotated المفقود من ملفات الـ labels الموجودة على القرص.
        يُدمج مع السجل الحالي (لا يكسر الموجود) ثم يُرفع إلى GitHub."""
        n_std = len(list((self.repo / LABELS_DIR).glob("*.txt"))) if (self.repo / LABELS_DIR).exists() else 0
        n_obb = len(list((self.repo / LABELS_OBB_DIR).glob("*.txt"))) if (self.repo / LABELS_OBB_DIR).exists() else 0
        n_labels = max(n_std, n_obb)
        if n_labels == 0:
            messagebox.showwarning(
                "لا ملفات labels",
                "لم يُعثر على ملفات labels في data/annotations.\n"
                "تأكد أنك على جذر الريبو وأن ملفات الـ labels موجودة.")
            return
        if not messagebox.askyesno(
                "استعادة الإحصائيات",
                f"سيُعاد بناء سجل الصور المعالَجة من {n_labels} ملف labels على القرص.\n\n"
                "• لن تُمَس الإدخالات الموجودة سلفاً في السجل (تُكمَّل حقولها الناقصة فقط).\n"
                "• الإدخالات المستعادة تُنسب إليك كـ annotated_by (الأصل غير معروف) وتُوسم recovered.\n"
                "• سيُحسب MD5 لكل صورة مقابلة (قد يستغرق دقائق لآلاف الصور).\n\n"
                "بعدها ستُدمج مع GitHub وتُرفع. متابعة؟"):
            return
        self.status.set("جارِ استعادة الإحصائيات من ملفات الـ labels…")
        self.root.update_idletasks()

        def worker():
            # نبدأ بأحدث سجل من GitHub حتى لا نطمس عمل زملاء تم تسجيله فعلاً
            self.lm.refresh_from_remote()

            def prog(done, total):
                self.root.after(0, lambda: self.status.set(
                    f"استعادة… {done}/{total} ملف label ({self.user})"))

            added, skipped, missing = recover_annotated_log(
                self.repo, self.lm.log, self.user, compute_hash=True, progress=prog)
            write_log(self.lm.log_path, self.lm.log)

            # رفع السجل المستعاد إلى GitHub (مع أي labels على القرص للاكتمال)
            ok, msg = self.lm.sync(
                f"recover(stats): rebuild {added} annotated entries from labels ({self.user})",
                extra_paths=[ANNOT_DIR])

            def done():
                if self.idx >= 0:
                    self._update_status()
                summary = (f"تمت الاستعادة:\n"
                           f"• أُضيف {added} إدخال جديد من ملفات الـ labels.\n"
                           f"• {skipped} إدخال كان موجوداً سلفاً (أُكمِلت حقوله الناقصة).\n")
                if missing:
                    ex = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
                    summary += (f"• {len(missing)} ملف label بلا صورة مطابقة في data/raw "
                                f"(سُجّل بدون MD5): {ex}\n")
                summary += "\n" + ("✅ " if ok else "⚠️ ") + msg.splitlines()[0]
                self.status.set(("✅ " if ok else "⚠️ ") +
                                f"استعادة: +{added} إدخال، {skipped} موجود مسبقاً")
                messagebox.showinfo("استعادة الإحصائيات", summary)
                if not ok:
                    messagebox.showwarning("مزامنة", msg)
            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def show_stats(self):
        # نجلب أحدث سجل من GitHub أولاً حتى تشمل الإحصائيات آخر عمل الفريق كله
        self.status.set("جارِ تحديث السجل من GitHub قبل عرض الإحصائيات…")
        self.root.update_idletasks()

        def worker():
            self.lm.refresh_from_remote()
            self.root.after(0, self._show_stats_window)

        threading.Thread(target=worker, daemon=True).start()

    def _show_stats_window(self):
        if self.idx >= 0:
            self._update_status()
        else:
            self.status.set("الإحصائيات محدّثة من GitHub")
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
            self.lm.refresh_from_remote()
            stats_path = export_stats_yaml(
                self.lm.log, self.repo / REPORTS_DIR / "annotation_stats.yaml")
            md = build_markdown_report(self.lm.log, len(self.images))
            msg = publish_report_pr(self.lm, md, stats_path)
            self.root.after(0, lambda: (self._update_status(),
                                        messagebox.showinfo("تقرير + PR", msg)))

        threading.Thread(target=worker, daemon=True).start()

    # ---------------------------------------------------------- مزامنة DVC --

    def sync_dvc(self, on_done=None):
        """مزامنة الصور الخام: سحب تحديثات الزملاء + رفع الصور الجديدة — في الخلفية.
        الـ labels لم تعد بحاجة لهذا الزر: تُرفع تلقائياً مع كل حفظ."""
        if not self.dvc.available():
            messagebox.showwarning("DVC", "أمر dvc غير متوفر في هذا الجهاز.\n"
                                          "ثبّته عبر:  pip install \"dvc[gdrive]\"")
            return
        if self.dvc_busy:
            self.status.set("عملية DVC جارية بالفعل — انتظر انتهاءها…")
            return
        self.dvc_busy = True
        self.dvc_btn.configure(state=tk.DISABLED)
        self.status.set("جارِ مزامنة الصور مع DVC… قد يستغرق دقائق حسب حجم التغييرات")
        self.root.update_idletasks()

        def worker():
            # نصنّف الحالة أولاً — ولا نُنفّذ سحباً (checkout) قبل التأكد أنه لا
            # توجد إضافات محلية، حتى لا تُحذف صور لم تُرفع بعد. نمرّر الحالة
            # المحسوبة لتفادي استدعاء dvc status إضافي (يقلّل فرص تصادم الأقفال).
            state = self.dvc.raw_state()
            if state in ("has_new", "unknown"):
                ok, msg = self.dvc.sync_raw(self.lm)          # رفع أولاً — يبدأ بـ dvc add
            else:  # missing (ناقصة، السحب آمن) أو clean (تحديث محتمل من الزملاء)
                ok, msg = self.dvc.pull_raw(state=state)

            def done():
                self.dvc_busy = False
                self.dvc_btn.configure(state=tk.NORMAL)
                self.status.set(("✅ " if ok else "⚠️ ") + msg.splitlines()[0])
                if not ok:
                    messagebox.showwarning("DVC", msg)
                if on_done:
                    on_done(ok)
            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------- إغلاق --

    def on_close(self):
        if not self._confirm_leave():
            return
        self._finalize_close()

    def _finalize_close(self):
        # تحرير كل أقفالي عند الخروج ومزامنتها (مع أي labels متبقية) حتى لا
        # تبقى صور محجوزة بلا داعٍ
        self.lm.release_all_mine()
        try:
            self.lm.sync(f"chore(locks): release locks on exit ({self.user})",
                         extra_paths=[ANNOT_DIR])
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
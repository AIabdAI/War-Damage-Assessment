"""
War-Damage-Assessment — Annotation, Statistics & Augmentation Pipeline
=======================================================================
Graduation Project — UCAS Gaza
Author  : Team / MLOps: Abdallah (AIabdAI)
Supervisor: Dr. Mohanad Abukmeil

Locking strategy (processed_log.json  ←  committed to git, NOT dvc):
─────────────────────────────────────────────────────────────────────
  On tool start   → git pull  (fetch latest locks)
  On image open   → lock current image  ┐
                  → claim next image    ┤ one git push
                  → git push            ┘
  On quit/finish  → release lock of last image + push

processed_log.json layout
──────────────────────────
{
  "annotated": {
    "img_001": { "hash": "...", "timestamp": "...", "boxes": 5,
                 "annotated_by": "Abdallah" }
  },
  "augmented": {
    "img_001": { "timestamp": "...", "num_augmented": 4, "outputs": [...] }
  },
  "locks": {
    "img_002": { "locked_by": "Abdallah", "locked_at": "...",
                 "expires_at": "..." }
  }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCIES (install once):
#   pip install opencv-python albumentations pillow pyyaml
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, json, hashlib, copy, random, datetime, argparse, subprocess
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional

import cv2
import numpy as np
import yaml as pyyaml
import albumentations as A

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES: List[str] = [
    "Brick_Wall",         # 0
    "Column",             # 1
    "Staircase",          # 2
    "Floor_Tiles",        # 3
    "Ceiling",            # 4
    "Beam",               # 5
    "Kitchen_Countertop", # 6
    "Sink",               # 7
    "Wall_Cabinet",       # 8
    "Chair",              # 9
    "Floor_Cabinet",      # 10
    "Bathtub",            # 11
    "Water_Faucet",       # 12
    "Window",             # 13
    "Fire_Extinguisher",  # 14
    "Roof",               # 15
    "Door",               # 16
    "Air_Conditioner",    # 17
    "Window_Sill",        # 18
    "Light_Fixture",      # 19
]
NUM_CLASSES = len(CLASS_NAMES)

PALETTE = [
    (255, 56,  56),  (255, 157,  51), (255, 225,  56), (99,  255,  60),
    (56,  255, 133), (56,  255, 221), (56, 166,  255), (56,  56,  255),
    (153,  56, 255), (255,  56, 200), (180, 180,  56), (56,  180, 180),
    (180,  56, 180), (255, 130, 130), (130, 255, 130), (130, 130, 255),
    (200, 200, 100), (100, 200, 200), (200, 100, 200), (255, 200,  50),
]

LOCK_TTL_HOURS  = 2          # lock expires after 2 hours of inactivity
LOG_PATH        = Path("processed_log.json")
GIT_AUTO_PUSH   = True       # set False to disable automatic git operations

# ══════════════════════════════════════════════════════════════════════════════
#  GIT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _git(args: List[str], silent: bool = True) -> Tuple[int, str]:
    """Run a git command. Returns (returncode, stdout+stderr)."""
    result = subprocess.run(
        ["git"] + args,
        capture_output=True, text=True
    )
    out = (result.stdout + result.stderr).strip()
    if not silent and out:
        print(f"  [git] {out}")
    return result.returncode, out


def git_pull() -> bool:
    print("[Git] Pulling latest lock state …")
    rc, out = _git(["pull", "--rebase", "--autostash"], silent=False)
    if rc != 0:
        print(f"  [Git] Pull failed (working offline): {out}")
        return False
    return True


def git_push_log(message: str) -> bool:
    """Stage processed_log.json + any new label files, then push."""
    if not GIT_AUTO_PUSH:
        return True
    _git(["add", str(LOG_PATH)])
    # also stage annotation labels if any
    _git(["add", "data/annotations/*.txt"])
    rc_commit, _ = _git(["commit", "-m", message, "--allow-empty"])
    rc_push,  out = _git(["push"], silent=False)
    if rc_push != 0:
        print(f"  [Git] Push failed: {out}")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESSED LOG  —  idempotency + distributed lock
# ══════════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _iso_to_dt(s: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(s.rstrip("Z"))


def _load_log() -> dict:
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text())
        except Exception:
            pass
    return {"annotated": {}, "augmented": {}, "locks": {},
            "class_counts": {n: 0 for n in CLASS_NAMES}}


def _save_log(log: dict) -> None:
    log.setdefault("locks", {})
    log.setdefault("class_counts", {n: 0 for n in CLASS_NAMES})
    LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False))


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


# ── Lock helpers ──────────────────────────────────────────────────────────────

def _is_lock_expired(entry: dict) -> bool:
    try:
        expires = _iso_to_dt(entry["expires_at"])
        return datetime.datetime.utcnow() > expires
    except Exception:
        return True


def _expire_stale_locks(log: dict) -> int:
    """Remove expired locks. Returns number removed."""
    stale = [s for s, e in log["locks"].items() if _is_lock_expired(e)]
    for s in stale:
        print(f"  [Lock] Expired lock removed: {s} (was held by {log['locks'][s].get('locked_by','?')})")
        del log["locks"][s]
    return len(stale)


def is_locked_by_other(stem: str, developer: str, log: dict) -> Optional[dict]:
    """
    Returns lock entry if the image is locked by someone else (and not expired).
    Returns None if free or locked by this developer.
    """
    entry = log["locks"].get(stem)
    if not entry:
        return None
    if _is_lock_expired(entry):
        return None
    if entry.get("locked_by") == developer:
        return None        # our own lock
    return entry


def acquire_lock(stem: str, developer: str, log: dict) -> None:
    """Write a lock for stem. Caller must call _save_log + git_push after."""
    now     = datetime.datetime.utcnow()
    expires = now + datetime.timedelta(hours=LOCK_TTL_HOURS)
    log["locks"][stem] = {
        "locked_by": developer,
        "locked_at": now.isoformat() + "Z",
        "expires_at": expires.isoformat() + "Z",
    }


def release_lock(stem: str, developer: str, log: dict) -> None:
    """Remove lock for stem if held by this developer."""
    entry = log["locks"].get(stem)
    if entry and entry.get("locked_by") == developer:
        del log["locks"][stem]


# ── Annotation / Augmentation state helpers ───────────────────────────────────

def is_annotated(stem: str, img_path: Path, log: dict) -> bool:
    entry = log["annotated"].get(stem)
    if not entry:
        return False
    return entry.get("hash") == _file_md5(img_path)


def mark_annotated(stem: str, img_path: Path,
                   num_boxes: int, developer: str, log: dict,
                   label_path: Optional[Path] = None) -> None:
    """
    Record annotation in log AND update live class_counts.
    label_path: the .txt file just saved — reads it to count per-class boxes.
    If re-annotating, subtracts the old counts first.
    """
    cc = log.setdefault("class_counts", {n: 0 for n in CLASS_NAMES})

    # ── Subtract old counts for this image (re-annotation case) ──────────
    old_entry = log["annotated"].get(stem)
    if old_entry and old_entry.get("class_boxes"):
        for cname, cnt in old_entry["class_boxes"].items():
            cc[cname] = max(0, cc.get(cname, 0) - cnt)

    # ── Count new boxes per class from the saved label file ───────────────
    per_class: dict = {n: 0 for n in CLASS_NAMES}
    if label_path and label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cid = int(parts[0])
                if 0 <= cid < NUM_CLASSES:
                    per_class[CLASS_NAMES[cid]] += 1

    # ── Add new counts to global totals ───────────────────────────────────
    for cname, cnt in per_class.items():
        cc[cname] = cc.get(cname, 0) + cnt

    log["annotated"][stem] = {
        "hash":          _file_md5(img_path),
        "timestamp":     _now_iso(),
        "boxes":         num_boxes,
        "annotated_by":  developer,
        "class_boxes":   per_class,      # per-image breakdown
    }


def is_augmented(stem: str, log: dict) -> bool:
    return stem in log["augmented"]


def mark_augmented(stem: str, num_aug: int,
                   outputs: List[str], log: dict) -> None:
    log["augmented"][stem] = {
        "timestamp":     _now_iso(),
        "num_augmented": num_aug,
        "outputs":       outputs,
    }


# ── Next-image selector ───────────────────────────────────────────────────────

def find_next_available(all_stems: List[str], current_stem: str,
                        developer: str, log: dict) -> Optional[str]:
    """
    Find the next image stem that is:
      - not yet annotated
      - not locked by another developer (or lock is expired)
      - not the current image
    Searches forward from current position, then wraps.
    """
    _expire_stale_locks(log)

    # order: stems after current first, then from beginning
    try:
        cur_idx = all_stems.index(current_stem)
    except ValueError:
        cur_idx = -1

    ordered = all_stems[cur_idx + 1:] + all_stems[:cur_idx + 1]

    for stem in ordered:
        if stem == current_stem:
            continue
        if is_annotated(stem,
                        # we only have stem here; annotated check uses hash
                        # pass a dummy path – annotation check will miss → OK
                        Path(stem), log):
            continue
        blocker = is_locked_by_other(stem, developer, log)
        if blocker:
            continue
        return stem
    return None          # all images done or locked


def print_log_summary(log: dict) -> None:
    ann   = log.get("annotated", {})
    aug   = log.get("augmented", {})
    locks = log.get("locks", {})
    active_locks = {s: e for s, e in locks.items() if not _is_lock_expired(e)}
    total_boxes  = sum(v.get("boxes", 0) for v in ann.values())
    print(f"\n{'─'*52}")
    print(f"  Processed Log Summary")
    print(f"{'─'*52}")
    print(f"  Annotated images  : {len(ann):>5}")
    print(f"  Total boxes       : {total_boxes:>5}")
    print(f"  Augmented images  : {len(aug):>5}")
    print(f"  Active locks      : {len(active_locks):>5}")
    if active_locks:
        for s, e in active_locks.items():
            print(f"    🔒 {s:<30} by {e['locked_by']}")
    print(f"{'─'*52}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — ANNOTATION TOOL  (Tkinter)
# ══════════════════════════════════════════════════════════════════════════════

def run_annotation_tool(images_dir: str, labels_dir: str,
                        developer: str,
                        skip_annotated: bool = True) -> None:
    """
    Tkinter YOLO annotation tool with distributed locking.

    On start     : git pull → expire stale locks → jump to first unclaimed image
    On save (s)  : mark annotated + release lock for current
                   + claim next image → ONE git push
    On next (n)  : save current → move to pre-claimed next image
    On quit (q)  : release current lock → push
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        sys.exit("tkinter not available — run in a desktop environment.")

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    all_image_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        for p in images_dir.glob(ext)
    )
    if not all_image_paths:
        print(f"[Annotation] No images found in {images_dir}")
        return

    # ── git pull to get latest locks ──────────────────────────────────────
    git_pull()
    log = _load_log()
    _expire_stale_locks(log)

    all_stems = [p.stem for p in all_image_paths]
    stem_to_path = {p.stem: p for p in all_image_paths}

    # ── Decide starting image ──────────────────────────────────────────────
    if skip_annotated:
        pending = [
            p for p in all_image_paths
            if not is_annotated(p.stem, p, log)
            and not is_locked_by_other(p.stem, developer, log)
        ]
        done = [
            p for p in all_image_paths
            if is_annotated(p.stem, p, log)
        ]
        image_paths = pending + done
        print(f"[Annotation] Developer: {developer}")
        print(f"  Pending : {len(pending)}  |  Done: {len(done)}")
    else:
        image_paths = all_image_paths

    if not image_paths:
        print("[Annotation] All images annotated or locked. Try again later.")
        return

    # ── Acquire lock on first image + claim next → push ───────────────────
    first_stem = image_paths[0].stem
    acquire_lock(first_stem, developer, log)

    next_stem  = find_next_available(all_stems, first_stem, developer, log)
    if next_stem:
        acquire_lock(next_stem, developer, log)
        print(f"  [Lock] Claimed: {first_stem}  +  pre-claimed: {next_stem}")
    else:
        print(f"  [Lock] Claimed: {first_stem}  (no next image available)")

    _save_log(log)
    git_push_log(f"[lock] {developer} claims {first_stem}" +
                 (f" + {next_stem}" if next_stem else ""))

    # ── Tkinter UI ────────────────────────────────────────────────────────
    state = {
        "idx":        0,
        "class_id":   0,
        "boxes":      [],
        "drawing":    False,
        "start_x":    0, "start_y": 0,
        "img_orig":   None,
        "img_h":      1, "img_w": 1,
        "scale":      1.0,
        "offset_x":   0, "offset_y": 0,
        "next_stem":  next_stem,   # pre-claimed next image
    }

    CANVAS_W, CANVAS_H = 1100, 780

    root = tk.Tk()
    root.title(f"War-Damage Annotation  —  {developer}")
    root.resizable(False, False)

    # Top bar
    top_frame = tk.Frame(root, bg="#2b2b2b")
    top_frame.pack(side=tk.TOP, fill=tk.X)

    lbl_file = tk.Label(top_frame, text="", font=("Courier", 10),
                        fg="#d4d4d4", bg="#2b2b2b", anchor="w")
    lbl_file.pack(side=tk.LEFT, padx=6)

    lbl_status = tk.Label(top_frame, text="", font=("Courier", 10, "bold"),
                          fg="#4ec9b0", bg="#2b2b2b", anchor="e")
    lbl_status.pack(side=tk.RIGHT, padx=6)

    canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H,
                       bg="#1e1e1e", cursor="crosshair")
    canvas.pack()

    bot_frame = tk.Frame(root, bg="#2b2b2b")
    bot_frame.pack(side=tk.BOTTOM, fill=tk.X)

    class_var = tk.IntVar(value=0)

    def on_class_change(*_):
        state["class_id"] = class_var.get()
        _refresh_status()

    class_combo = ttk.Combobox(
        bot_frame,
        values=[f"{i}: {n}" for i, n in enumerate(CLASS_NAMES)],
        state="readonly", width=28, textvariable=class_var,
    )
    class_combo.current(0)
    class_combo.bind("<<ComboboxSelected>>", on_class_change)
    class_combo.pack(side=tk.LEFT, padx=4, pady=4)

    tk.Label(bot_frame,
             text="[n] Next  [p] Prev  [s] Save  [Del] Remove Last  [q] Quit",
             font=("Arial", 9), fg="#a0a0a0", bg="#2b2b2b"
             ).pack(side=tk.LEFT, padx=10)

    # ── Helpers ────────────────────────────────────────────────────────────
    def label_path_of(img_path: Path) -> Path:
        return labels_dir / (img_path.stem + ".txt")

    def _refresh_status():
        idx   = state["idx"]
        path  = image_paths[idx]
        done  = is_annotated(path.stem, path, log)
        lock  = log["locks"].get(path.stem, {})
        cid   = state["class_id"]
        boxes = len(state["boxes"])
        nxt   = state.get("next_stem") or "—"

        if done:
            flag = "✔ DONE"
        elif lock.get("locked_by") == developer:
            flag = "🔒 LOCKED (you)"
        else:
            flag = "○ PENDING"

        lbl_status.config(
            text=f"{flag}  |  {CLASS_NAMES[cid]}  |  boxes:{boxes}  |  next→{nxt}",
            fg="#4ec9b0" if done else "#f48771"
        )

    def load_image(idx: int):
        path = image_paths[idx]
        img  = cv2.imread(str(path))
        if img is None:
            return
        state["img_orig"] = img
        h, w = img.shape[:2]
        state["img_h"], state["img_w"] = h, w
        scale             = min(CANVAS_W / w, CANVAS_H / h, 1.0)
        state["scale"]    = scale
        state["offset_x"] = int((CANVAS_W - w * scale) / 2)
        state["offset_y"] = int((CANVAS_H - h * scale) / 2)

        lp = label_path_of(path)
        state["boxes"] = []
        if lp.exists():
            for line in lp.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                    state["boxes"].append([cid, xc, yc, bw, bh])

        done_tag = "✔" if is_annotated(path.stem, path, log) else "🔒"
        lbl_file.config(
            text=f"[{idx+1}/{len(image_paths)}] {done_tag}  {path.name}"
        )
        _refresh_status()
        redraw()

    def norm_to_canvas(xc, yc, bw, bh):
        s, ox, oy = state["scale"], state["offset_x"], state["offset_y"]
        W, H      = state["img_w"], state["img_h"]
        x1 = int((xc - bw / 2) * W * s + ox)
        y1 = int((yc - bh / 2) * H * s + oy)
        x2 = int((xc + bw / 2) * W * s + ox)
        y2 = int((yc + bh / 2) * H * s + oy)
        return x1, y1, x2, y2

    def canvas_to_norm(cx1, cy1, cx2, cy2):
        s, ox, oy = state["scale"], state["offset_x"], state["offset_y"]
        W, H      = state["img_w"], state["img_h"]
        x1 = (min(cx1, cx2) - ox) / (W * s)
        y1 = (min(cy1, cy2) - oy) / (H * s)
        x2 = (max(cx1, cx2) - ox) / (W * s)
        y2 = (max(cy1, cy2) - oy) / (H * s)
        xc = (x1 + x2) / 2;  yc = (y1 + y2) / 2
        bw = x2 - x1;        bh = y2 - y1
        return (max(0., min(1., v)) for v in (xc, yc, bw, bh))

    def redraw():
        canvas.delete("all")
        img = state["img_orig"]
        if img is None:
            return
        s, ox, oy = state["scale"], state["offset_x"], state["offset_y"]
        disp      = cv2.resize(img, None, fx=s, fy=s,
                               interpolation=cv2.INTER_AREA)
        disp_rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        pil_img   = Image.fromarray(disp_rgb)
        tk_img    = ImageTk.PhotoImage(pil_img)
        canvas._tk_img = tk_img
        canvas.create_image(ox, oy, anchor="nw", image=tk_img)
        for box in state["boxes"]:
            cid, xc, yc, bw, bh = box
            x1, y1, x2, y2 = norm_to_canvas(xc, yc, bw, bh)
            r, g, b = PALETTE[cid % len(PALETTE)]
            color   = f"#{r:02x}{g:02x}{b:02x}"
            canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            canvas.create_text(x1 + 3, y1 - 9, text=CLASS_NAMES[cid],
                               fill=color, anchor="w",
                               font=("Arial", 8, "bold"))

    def _save_current_and_advance():
        """
        Core atomic operation:
          1. Write label file
          2. Mark annotated in log
          3. Release lock on current image
          4. Move to pre-claimed next image (already locked)
          5. Claim the image AFTER that as new next
          6. ONE git push
        """
        idx  = state["idx"]
        path = image_paths[idx]
        stem = path.stem

        # 1. Write label
        lp    = label_path_of(path)
        lines = [
            f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
            for b in state["boxes"]
        ]
        lp.write_text("\n".join(lines))

        # 2. Mark annotated
        mark_annotated(stem, path, len(state["boxes"]), developer, log)

        # 3. Release current lock
        release_lock(stem, developer, log)

        # 4. Determine next image (was pre-claimed)
        claimed_next = state.get("next_stem")

        # 5. Pre-claim the image after that
        new_next = find_next_available(all_stems, claimed_next or stem,
                                       developer, log)
        if new_next:
            acquire_lock(new_next, developer, log)

        state["next_stem"] = new_next
        _save_log(log)

        # 6. One push
        msg = f"[annotate] {developer}: done={stem}"
        if claimed_next:
            msg += f" | active={claimed_next}"
        if new_next:
            msg += f" | next={new_next}"
        git_push_log(msg)

        print(f"  [Saved] {path.name}  ({len(state['boxes'])} boxes)"
              f"  →  next: {claimed_next or '—'}")

        # Return the path of the image to navigate to
        return stem_to_path.get(claimed_next) if claimed_next else None

    def save_and_stay():
        """Save without advancing (s key)."""
        idx  = state["idx"]
        path = image_paths[idx]
        stem = path.stem
        lp   = label_path_of(path)
        lines = [
            f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
            for b in state["boxes"]
        ]
        lp.write_text("\n".join(lines))
        # Mark but keep lock — will fully release on n/q
        mark_annotated(stem, path, len(state["boxes"]), developer, log)
        _save_log(log)
        _refresh_status()
        print(f"  [Saved] {path.name}  ({len(state['boxes'])} boxes) — lock kept")

    # ── Mouse events ───────────────────────────────────────────────────────
    _drag_id = [None]

    def on_press(e):
        state["drawing"] = True
        state["start_x"] = e.x;  state["start_y"] = e.y

    def on_drag(e):
        if not state["drawing"]:
            return
        if _drag_id[0]:
            canvas.delete(_drag_id[0])
        cid    = state["class_id"]
        r, g, b = PALETTE[cid % len(PALETTE)]
        color  = f"#{r:02x}{g:02x}{b:02x}"
        _drag_id[0] = canvas.create_rectangle(
            state["start_x"], state["start_y"], e.x, e.y,
            outline=color, width=2, dash=(4, 2)
        )

    def on_release(e):
        if not state["drawing"]:
            return
        state["drawing"] = False
        if _drag_id[0]:
            canvas.delete(_drag_id[0]);  _drag_id[0] = None
        x1, y1 = state["start_x"], state["start_y"]
        x2, y2 = e.x, e.y
        if abs(x2 - x1) < 8 or abs(y2 - y1) < 8:
            return
        xc, yc, bw, bh = canvas_to_norm(x1, y1, x2, y2)
        if bw > 0.005 and bh > 0.005:
            state["boxes"].append([state["class_id"], xc, yc, bw, bh])
        _refresh_status();  redraw()

    def on_right_click(e):
        for i, box in enumerate(state["boxes"]):
            cid, xc, yc, bw, bh = box
            x1, y1, x2, y2 = norm_to_canvas(xc, yc, bw, bh)
            if x1 <= e.x <= x2 and y1 <= e.y <= y2:
                state["boxes"].pop(i);  break
        _refresh_status();  redraw()

    canvas.bind("<ButtonPress-1>",   on_press)
    canvas.bind("<B1-Motion>",       on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    canvas.bind("<Button-3>",        on_right_click)

    # ── Key events ─────────────────────────────────────────────────────────
    def on_key(e):
        k = e.keysym.lower()

        if k == "n":
            # Save + advance to pre-claimed next image
            next_path = _save_current_and_advance()
            if next_path and next_path in image_paths:
                state["idx"] = image_paths.index(next_path)
            else:
                # fallback: just move forward in list
                state["idx"] = min(state["idx"] + 1, len(image_paths) - 1)
            load_image(state["idx"])

        elif k == "p":
            save_and_stay()
            state["idx"] = max(state["idx"] - 1, 0)
            load_image(state["idx"])

        elif k == "s":
            save_and_stay()

        elif k == "q":
            # Release current lock + push, then quit
            cur_stem = image_paths[state["idx"]].stem
            save_and_stay()
            release_lock(cur_stem, developer, log)
            # Also release pre-claimed next if we're quitting
            nxt = state.get("next_stem")
            if nxt:
                release_lock(nxt, developer, log)
            _save_log(log)
            git_push_log(f"[release] {developer} released locks on quit")
            print_log_summary(log)
            root.destroy()

        elif k in ("delete", "backspace"):
            if state["boxes"]:
                state["boxes"].pop()
            _refresh_status();  redraw()

        elif k.isdigit():
            cid = int(k)
            if cid < NUM_CLASSES:
                state["class_id"] = cid
                class_combo.current(cid)
                _refresh_status()

    root.bind("<Key>", on_key)
    load_image(0)
    root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — STATISTICS  →  YAML  +  GitHub CML Report
# ══════════════════════════════════════════════════════════════════════════════

def compute_statistics(labels_dir: str,
                       output_yaml: str = "annotation_stats.yaml") -> dict:
    labels_dir = Path(labels_dir)
    stats = {
        "generated_at": _now_iso(),
        "total_images":  0,
        "total_boxes":   0,
        "classes":       {},
    }

    class_counts     = defaultdict(int)
    class_images     = defaultdict(set)
    widths_by_class  = defaultdict(list)
    heights_by_class = defaultdict(list)
    images_with_ann  = 0

    label_files = sorted(labels_dir.glob("*.txt"))
    stats["total_images"] = len(label_files)

    for lf in label_files:
        lines = [l.strip() for l in lf.read_text().splitlines() if l.strip()]
        if lines:
            images_with_ann += 1
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            bw, bh = float(parts[3]), float(parts[4])
            class_counts[cid] += 1
            class_images[cid].add(lf.stem)
            widths_by_class[cid].append(bw)
            heights_by_class[cid].append(bh)
            stats["total_boxes"] += 1

    stats["images_with_annotation"] = images_with_ann

    for cid, name in enumerate(CLASS_NAMES):
        count = class_counts.get(cid, 0)
        imgs  = len(class_images.get(cid, set()))
        ws    = widths_by_class.get(cid, [0])
        hs    = heights_by_class.get(cid, [0])
        pct   = round(count / max(stats["total_boxes"], 1) * 100, 2)
        stats["classes"][name] = {
            "class_id":          cid,
            "total_boxes":       count,
            "images_present":    imgs,
            "percentage_of_all": pct,
            "avg_bbox_width":    round(float(np.mean(ws)), 4) if ws else 0.0,
            "avg_bbox_height":   round(float(np.mean(hs)), 4) if hs else 0.0,
        }

    out_path = Path(output_yaml)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        pyyaml.dump(stats, f, allow_unicode=True, sort_keys=False)
    print(f"[Stats] Saved → {out_path}")
    return stats


def generate_cml_report(stats: dict,
                        report_path: str = "annotation_report.md") -> str:
    log = _load_log()
    active_locks = {s: e for s, e in log.get("locks", {}).items()
                    if not _is_lock_expired(e)}
    lines = [
        "# 📊 Annotation Statistics Report",
        f"> Generated: `{stats['generated_at']}`",
        "",
        "## Summary",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total images | **{stats['total_images']}** |",
        f"| Images annotated | **{stats.get('images_with_annotation', '–')}** |",
        f"| Total boxes | **{stats['total_boxes']}** |",
        f"| Augmented | **{len(log.get('augmented', {}))}** |",
        f"| Active locks | **{len(active_locks)}** |",
        "",
    ]

    if active_locks:
        lines += ["## 🔒 Active Locks", "| Image | Developer | Expires |",
                  "|-------|-----------|---------|"]
        for s, e in active_locks.items():
            lines.append(f"| `{s}` | {e['locked_by']} | {e['expires_at']} |")
        lines.append("")

    lines += [
        "## Per-Class Breakdown",
        "| # | Class | Boxes | % | Images | Avg W | Avg H |",
        "|---|-------|------:|--:|-------:|------:|------:|",
    ]
    for name, info in stats["classes"].items():
        lines.append(
            f"| {info['class_id']:>2} | `{name}` | {info['total_boxes']:>5} "
            f"| {info['percentage_of_all']:>5}% | {info['images_present']:>5} "
            f"| {info['avg_bbox_width']:.3f} | {info['avg_bbox_height']:.3f} |"
        )

    max_count = max((v["total_boxes"] for v in stats["classes"].values()), default=1)
    lines += ["", "## Distribution", "```"]
    for name, info in stats["classes"].items():
        bar = "█" * int(info["total_boxes"] / max(max_count, 1) * 40)
        lines.append(f"{name:<25} {bar:<40} {info['total_boxes']}")
    lines += ["```", ""]

    report = "\n".join(lines)
    Path(report_path).write_text(report, encoding="utf-8")
    print(f"[Report] Saved → {report_path}")
    return report


def publish_cml_report(report_path: str = "annotation_report.md") -> None:
    try:
        result = subprocess.run(
            ["cml", "comment", "create", report_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("[CML] Report posted ✓")
        else:
            print(f"[CML] Error: {result.stderr}")
    except FileNotFoundError:
        print("[CML] cml not found — install: npm install -g @dvcorg/cml")


# ══════════════════════════════════════════════════════════════════════════════
#  PART 3 — AUGMENTATION  (bbox-aware + idempotent)
# ══════════════════════════════════════════════════════════════════════════════

def _read_yolo_label(label_path: Path, img_w: int, img_h: int) -> List:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])
        boxes.append([cid,
                      (xc - bw/2)*img_w, (yc - bh/2)*img_h,
                      (xc + bw/2)*img_w, (yc + bh/2)*img_h])
    return boxes


def _write_yolo_label(label_path: Path, boxes: List,
                      img_w: int, img_h: int) -> None:
    lines = []
    for box in boxes:
        cid, x1, y1, x2, y2 = box
        xc = (x1+x2)/2/img_w;  yc = (y1+y2)/2/img_h
        bw = (x2-x1)/img_w;    bh = (y2-y1)/img_h
        xc = max(0., min(1., xc));  yc = max(0., min(1., yc))
        bw = max(0.001, min(1., bw)); bh = max(0.001, min(1., bh))
        lines.append(f"{int(cid)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines))


def build_albumentations_pipeline(img_size: int = 640) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomResizedCrop(size=(img_size, img_size),
                            scale=(0.65, 1.0), ratio=(0.75, 1.33), p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.3, hue=0.05, p=0.5),
        A.GaussNoise(std_range=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    ], bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["class_ids"],
        min_visibility=0.2, clip=True,
    ))


def mosaic_augment(image_paths, label_paths, img_size=640):
    assert len(image_paths) >= 4
    indices = random.sample(range(len(image_paths)), 4)
    cy, cx  = img_size // 2, img_size // 2
    mosaic  = np.full((img_size*2, img_size*2, 3), 114, dtype=np.uint8)
    final_boxes = []
    placements  = [(0,0,cy,cx),(0,cx,cy,img_size),
                   (cy,0,img_size,cx),(cy,cx,img_size,img_size)]
    for i, idx in enumerate(indices):
        img = cv2.imread(str(image_paths[idx]))
        if img is None:
            img = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
        h, w = img.shape[:2]
        y1p,x1p,y2p,x2p = placements[i]
        ph,pw = y2p-y1p, x2p-x1p
        mosaic[y1p:y2p, x1p:x2p] = cv2.resize(img, (pw, ph))
        for box in _read_yolo_label(label_paths[idx], w, h):
            cid,bx1,by1,bx2,by2 = box
            sx,sy = pw/w, ph/h
            final_boxes.append([cid, bx1*sx+x1p, by1*sy+y1p,
                                      bx2*sx+x1p, by2*sy+y1p])
    crop = mosaic[cy-img_size//2:cy+img_size//2,
                  cx-img_size//2:cx+img_size//2]
    shifted = []
    for box in final_boxes:
        cid,bx1,by1,bx2,by2 = box
        bx1-=cx-img_size//2; bx2-=cx-img_size//2
        by1-=cy-img_size//2; by2-=cy-img_size//2
        bx1=max(0.,bx1); by1=max(0.,by1)
        bx2=min(img_size,bx2); by2=min(img_size,by2)
        if bx2-bx1>2 and by2-by1>2:
            shifted.append([cid,bx1,by1,bx2,by2])
    return crop, shifted


def mixup_augment(img1, boxes1, img2, boxes2, alpha=0.5):
    lam   = random.betavariate(alpha, alpha)
    h, w  = img1.shape[:2]
    img2r = cv2.resize(img2, (w, h))
    mixed = (lam*img1.astype(np.float32) +
             (1-lam)*img2r.astype(np.float32)).clip(0,255).astype(np.uint8)
    h2,w2 = img2.shape[:2]
    scaled2 = [[b[0], b[1]*w/w2, b[2]*h/h2, b[3]*w/w2, b[4]*h/h2]
               for b in boxes2]
    return mixed, boxes1 + scaled2


def augment_dataset(images_dir, labels_dir, output_images_dir,
                    output_labels_dir, num_augmented_per_image=4,
                    img_size=640, enable_mosaic=True, enable_mixup=True,
                    mosaic_prob=0.3, mixup_prob=0.2, force=False):
    images_dir  = Path(images_dir)
    labels_dir  = Path(labels_dir)
    out_img_dir = Path(output_images_dir)
    out_lbl_dir = Path(output_labels_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    all_img = sorted(p for ext in ("*.jpg","*.jpeg","*.png","*.bmp")
                     for p in images_dir.glob(ext))
    all_lbl = [labels_dir / (p.stem + ".txt") for p in all_img]

    if not all_img:
        print(f"[Augment] No images in {images_dir}"); return

    log = _load_log()
    image_paths, label_paths, skipped = [], [], 0

    for ip, lp in zip(all_img, all_lbl):
        if not is_annotated(ip.stem, ip, log):
            skipped += 1; continue
        if not force and is_augmented(ip.stem, log):
            skipped += 1; continue
        image_paths.append(ip); label_paths.append(lp)

    print(f"[Augment] {len(image_paths)} to process | {skipped} skipped")
    if not image_paths:
        print("[Augment] Nothing to do. Use --force to re-augment."); return

    transform = build_albumentations_pipeline(img_size)
    total_out = 0

    for img_idx, (img_path, lbl_path) in enumerate(zip(image_paths, label_paths)):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] {img_path} unreadable"); continue
        img = cv2.resize(img, (img_size, img_size))
        h, w = img.shape[:2]
        boxes_raw = _read_yolo_label(lbl_path, w, h)
        stem = img_path.stem
        outputs = []

        orig_out = out_img_dir / f"{stem}_orig.jpg"
        cv2.imwrite(str(orig_out), img)
        _write_yolo_label(out_lbl_dir / f"{stem}_orig.txt", boxes_raw, w, h)
        outputs.append(f"{stem}_orig.jpg"); total_out += 1

        for aug_i in range(num_augmented_per_image):
            chosen_img   = img.copy()
            chosen_boxes = copy.deepcopy(boxes_raw)

            if enable_mosaic and random.random() < mosaic_prob and len(image_paths) >= 4:
                chosen_img, chosen_boxes = mosaic_augment(image_paths, label_paths, img_size)
                chosen_img = cv2.resize(chosen_img, (img_size, img_size))
            elif enable_mixup and random.random() < mixup_prob and len(image_paths) >= 2:
                oi  = random.randint(0, len(image_paths)-1)
                oi2 = cv2.imread(str(image_paths[oi]))
                if oi2 is not None:
                    oi2 = cv2.resize(oi2, (img_size, img_size))
                    ob  = _read_yolo_label(label_paths[oi], img_size, img_size)
                    chosen_img, chosen_boxes = mixup_augment(chosen_img, chosen_boxes, oi2, ob)

            bboxes_alb = [[b[1],b[2],b[3],b[4]] for b in chosen_boxes]
            class_ids  = [b[0] for b in chosen_boxes]
            try:
                if bboxes_alb:
                    r = transform(image=chosen_img, bboxes=bboxes_alb, class_ids=class_ids)
                    aug_img   = r["image"]
                    new_boxes = [[cid,*box] for cid,box in zip(r["class_ids"], r["bboxes"])]
                else:
                    r = transform(image=chosen_img, bboxes=[], class_ids=[])
                    aug_img = r["image"]; new_boxes = []
            except Exception as ex:
                print(f"  [WARN] {img_path.name}: {ex}")
                aug_img = chosen_img; new_boxes = chosen_boxes

            out_stem = f"{stem}_aug{aug_i:02d}"
            cv2.imwrite(str(out_img_dir / f"{out_stem}.jpg"), aug_img)
            ah, aw = aug_img.shape[:2]
            _write_yolo_label(out_lbl_dir / f"{out_stem}.txt", new_boxes, aw, ah)
            outputs.append(f"{out_stem}.jpg"); total_out += 1

        mark_augmented(stem, num_augmented_per_image, outputs, log)
        if (img_idx+1) % 20 == 0 or (img_idx+1) == len(image_paths):
            print(f"  [{img_idx+1}/{len(image_paths)}] done …")

    _save_log(log)
    print(f"[Augment] Done. {total_out} files → {out_img_dir}")
    print_log_summary(log)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="War-Damage-Assessment Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    ann = sub.add_parser("annotate", help="Launch annotation tool")
    ann.add_argument("--images",     default="data/raw")
    ann.add_argument("--labels",     default="data/annotations")
    ann.add_argument("--dev",        required=True,
                     help="Developer name (e.g. Abdallah)")
    ann.add_argument("--reannotate", action="store_true")

    st = sub.add_parser("stats", help="Compute annotation statistics")
    st.add_argument("--labels",  default="data/annotations")
    st.add_argument("--yaml",    default="annotation_stats.yaml")
    st.add_argument("--report",  default="annotation_report.md")
    st.add_argument("--publish", action="store_true")

    ag = sub.add_parser("augment", help="Run augmentation pipeline")
    ag.add_argument("--images",     default="data/raw")
    ag.add_argument("--labels",     default="data/annotations")
    ag.add_argument("--out-images", default="data/processed/images")
    ag.add_argument("--out-labels", default="data/processed/labels")
    ag.add_argument("--n",          type=int, default=4)
    ag.add_argument("--size",       type=int, default=640)
    ag.add_argument("--no-mosaic",  action="store_true")
    ag.add_argument("--no-mixup",   action="store_true")
    ag.add_argument("--force",      action="store_true")

    fp = sub.add_parser("full", help="Stats + augmentation")
    fp.add_argument("--images",     default="data/raw")
    fp.add_argument("--labels",     default="data/annotations")
    fp.add_argument("--out-images", default="data/processed/images")
    fp.add_argument("--out-labels", default="data/processed/labels")
    fp.add_argument("--yaml",       default="annotation_stats.yaml")
    fp.add_argument("--report",     default="annotation_report.md")
    fp.add_argument("--n",          type=int, default=4)
    fp.add_argument("--size",       type=int, default=640)
    fp.add_argument("--publish",    action="store_true")
    fp.add_argument("--force",      action="store_true")

    sub.add_parser("log", help="Print processed_log.json summary")

    return p.parse_args()


def main():
    args = parse_args()

    if args.command == "annotate":
        run_annotation_tool(
            args.images, args.labels,
            developer=args.dev,
            skip_annotated=not args.reannotate,
        )
    elif args.command == "stats":
        stats = compute_statistics(args.labels, args.yaml)
        generate_cml_report(stats, args.report)
        if args.publish:
            publish_cml_report(args.report)
    elif args.command == "augment":
        augment_dataset(
            args.images, args.labels,
            args.out_images, args.out_labels,
            args.n, args.size,
            not args.no_mosaic, not args.no_mixup,
            force=args.force,
        )
    elif args.command == "full":
        stats = compute_statistics(args.labels, args.yaml)
        generate_cml_report(stats, args.report)
        if args.publish:
            publish_cml_report(args.report)
        augment_dataset(
            args.images, args.labels,
            args.out_images, args.out_labels,
            args.n, args.size, force=args.force,
        )
    elif args.command == "log":
        print_log_summary(_load_log())


if __name__ == "__main__":
    main()
"""
War-Damage-Assessment — Pipeline GUI Launcher
==============================================
Graduation Project — UCAS Gaza
Author: Abdallah (AIabdAI)

Tkinter dashboard to run all pipeline commands without the terminal.
Requires pipeline.py in the same directory.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess, threading, sys, json, datetime
from pathlib import Path

LOG_PATH = Path("processed_log.json")
SCRIPT   = Path(__file__).parent / "pipeline.py"

# ── Colour palette (dark theme) ──────────────────────────────────────────────
BG       = "#1a1a2e"   # deep navy — base
SURFACE  = "#16213e"   # card surface
SURFACE2 = "#0f3460"   # raised card
ACCENT   = "#e94560"   # coral red — signature colour (war-damage theme)
ACCENT2  = "#533483"   # purple
TEXT     = "#eaeaea"
MUTED    = "#8892a4"
SUCCESS  = "#4caf91"
WARNING  = "#f4a261"
DANGER   = "#e94560"
BORDER   = "#2a2a4a"

FONT_H1  = ("Segoe UI", 18, "bold")
FONT_H2  = ("Segoe UI", 13, "bold")
FONT_H3  = ("Segoe UI", 11, "bold")
FONT_B   = ("Segoe UI", 10)
FONT_M   = ("Consolas", 9)


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _load_log() -> dict:
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text())
        except Exception:
            pass
    return {"annotated": {}, "augmented": {}, "locks": {}}


def _is_expired(entry: dict) -> bool:
    try:
        exp = datetime.datetime.fromisoformat(entry["expires_at"].rstrip("Z"))
        return datetime.datetime.utcnow() > exp
    except Exception:
        return True


def _run_cmd(cmd: list, log_widget: scrolledtext.ScrolledText,
             on_done=None):
    """Run a subprocess and stream output to the log widget."""
    def target():
        log_widget.config(state="normal")
        log_widget.insert("end", f"\n$ {' '.join(cmd)}\n", "cmd")
        log_widget.see("end")
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            for line in proc.stdout:
                log_widget.insert("end", line)
                log_widget.see("end")
            proc.wait()
            rc = proc.returncode
            tag = "ok" if rc == 0 else "err"
            log_widget.insert("end",
                f"[{'OK' if rc == 0 else 'FAILED'}  exit {rc}]\n", tag)
        except Exception as e:
            log_widget.insert("end", f"[ERROR] {e}\n", "err")
        log_widget.see("end")
        log_widget.config(state="disabled")
        if on_done:
            log_widget.after(0, on_done)
    threading.Thread(target=target, daemon=True).start()


# ════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════════════════════════════════════

class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("War-Damage Assessment — Pipeline Control")
        self.geometry("1060x720")
        self.resizable(True, True)
        self.configure(bg=BG)

        self._build_styles()
        self._build_layout()
        self._refresh_stats()

        # Auto-refresh stats every 5 s
        self._schedule_refresh()

    # ── Styles ────────────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame",       background=BG)
        s.configure("Card.TFrame",  background=SURFACE,  relief="flat")
        s.configure("Card2.TFrame", background=SURFACE2, relief="flat")
        s.configure("TLabel",       background=BG,       foreground=TEXT,
                    font=FONT_B)
        s.configure("H1.TLabel",    background=BG,       foreground=TEXT,
                    font=FONT_H1)
        s.configure("H2.TLabel",    background=SURFACE,  foreground=TEXT,
                    font=FONT_H2)
        s.configure("H2b.TLabel",   background=BG,       foreground=TEXT,
                    font=FONT_H2)
        s.configure("Muted.TLabel", background=SURFACE,  foreground=MUTED,
                    font=("Segoe UI", 9))
        s.configure("Muted2.TLabel",background=BG,       foreground=MUTED,
                    font=("Segoe UI", 9))
        s.configure("Stat.TLabel",  background=SURFACE,  foreground=TEXT,
                    font=("Segoe UI", 26, "bold"))
        s.configure("Tag.TLabel",   background=ACCENT2,  foreground=TEXT,
                    font=("Segoe UI", 8, "bold"), padding=(6, 2))
        s.configure("OK.TLabel",    background=SURFACE,  foreground=SUCCESS,
                    font=("Segoe UI", 9, "bold"))
        s.configure("Warn.TLabel",  background=SURFACE,  foreground=WARNING,
                    font=("Segoe UI", 9, "bold"))
        # Buttons
        s.configure("TButton", font=FONT_B, relief="flat",
                    background=SURFACE2, foreground=TEXT,
                    borderwidth=0, padding=(14, 7))
        s.map("TButton",
              background=[("active", ACCENT2), ("pressed", ACCENT)],
              foreground=[("active", TEXT)])
        s.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                    font=("Segoe UI", 10, "bold"), padding=(16, 8))
        s.map("Accent.TButton",
              background=[("active", "#c73652"), ("pressed", "#a02840")])
        s.configure("TEntry",  fieldbackground=SURFACE2, foreground=TEXT,
                    insertcolor=TEXT, borderwidth=0, padding=(6, 4))
        s.configure("TCombobox", fieldbackground=SURFACE2, foreground=TEXT,
                    background=SURFACE2, selectbackground=ACCENT2,
                    borderwidth=0)
        s.configure("TCheckbutton", background=BG, foreground=TEXT,
                    font=FONT_B)
        s.map("TCheckbutton", background=[("active", BG)])
        s.configure("TSeparator", background=BORDER)

    # ── Layout ────────────────────────────────────────────────────────────
    def _build_layout(self):
        # Header
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", padx=20, pady=(18, 0))
        ttk.Label(hdr, text="⬛ War-Damage Assessment",
                  style="H1.TLabel").pack(side="left")
        ttk.Label(hdr, text="Pipeline Control Dashboard",
                  style="Muted2.TLabel").pack(side="left", padx=(12, 0),
                                              pady=(6, 0))
        self._btn_refresh = ttk.Button(hdr, text="⟳  Refresh",
                                       command=self._refresh_stats)
        self._btn_refresh.pack(side="right")

        ttk.Separator(self).pack(fill="x", padx=20, pady=10)

        # Two-column body
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 8))
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        left  = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")

        self._build_left(left)
        self._build_right(right)

    # ── LEFT: stats + controls ────────────────────────────────────────────
    def _build_left(self, parent):
        parent.rowconfigure(3, weight=1)

        # ── Stats cards ───────────────────────────────────────────────────
        ttk.Label(parent, text="Overview", style="H2b.TLabel").pack(
            anchor="w", pady=(0, 6))

        stat_row = ttk.Frame(parent)
        stat_row.pack(fill="x", pady=(0, 10))
        for i in range(3):
            stat_row.columnconfigure(i, weight=1)

        self._stat_ann   = self._stat_card(stat_row, "Annotated",  "0", 0)
        self._stat_aug   = self._stat_card(stat_row, "Augmented",  "0", 1)
        self._stat_locks = self._stat_card(stat_row, "Active locks","0", 2)

        # ── Locks list ────────────────────────────────────────────────────
        locks_card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        locks_card.pack(fill="x", pady=(0, 10))
        ttk.Label(locks_card, text="Active locks", style="H2.TLabel").pack(
            anchor="w")
        self._locks_box = tk.Text(locks_card, height=4, bg=SURFACE2,
                                  fg=TEXT, font=FONT_M, relief="flat",
                                  state="disabled", bd=0)
        self._locks_box.pack(fill="x", pady=(6, 0))

        ttk.Separator(parent).pack(fill="x", pady=8)

        # ── Common paths ──────────────────────────────────────────────────
        ttk.Label(parent, text="Paths", style="H2b.TLabel").pack(
            anchor="w", pady=(0, 6))

        paths_card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        paths_card.pack(fill="x", pady=(0, 10))

        self.var_images  = self._path_row(paths_card, "Raw images",    "data/raw")
        self.var_labels  = self._path_row(paths_card, "Annotations",   "data/annotations")
        self.var_out_img = self._path_row(paths_card, "Processed img", "data/processed/images")
        self.var_out_lbl = self._path_row(paths_card, "Processed lbl", "data/processed/labels")

        ttk.Separator(parent).pack(fill="x", pady=8)

        # ── Developer name ────────────────────────────────────────────────
        dev_frame = ttk.Frame(parent)
        dev_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(dev_frame, text="Developer name",
                  style="Muted2.TLabel").pack(side="left")
        self.var_dev = tk.StringVar(value="Abdallah")
        ttk.Entry(dev_frame, textvariable=self.var_dev, width=18).pack(
            side="right")

    def _stat_card(self, parent, label, init, col):
        f = ttk.Frame(parent, style="Card.TFrame", padding=(10, 8))
        f.grid(row=0, column=col, sticky="ew", padx=(0, 6) if col < 2 else 0)
        ttk.Label(f, text=label, style="Muted.TLabel").pack(anchor="w")
        var = tk.StringVar(value=init)
        ttk.Label(f, textvariable=var, style="Stat.TLabel").pack(anchor="w")
        return var

    def _path_row(self, parent, label: str, default: str) -> tk.StringVar:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=f"{label}:", style="Muted.TLabel",
                  width=14).pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Entry(row, textvariable=var, font=FONT_M).pack(
            side="left", fill="x", expand=True, padx=(4, 4))
        ttk.Button(row, text="…", width=2,
                   command=lambda v=var: self._browse(v)).pack(side="right")
        return var

    def _browse(self, var: tk.StringVar):
        d = filedialog.askdirectory(initialdir=var.get() or ".")
        if d:
            var.set(d)

    # ── RIGHT: tabs ───────────────────────────────────────────────────────
    def _build_right(self, parent):
        parent.rowconfigure(1, weight=1)

        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        # Style notebook tabs
        s = ttk.Style()
        s.configure("TNotebook",          background=BG,      borderwidth=0)
        s.configure("TNotebook.Tab",      background=SURFACE, foreground=MUTED,
                    padding=(14, 6), font=FONT_B)
        s.map("TNotebook.Tab",
              background=[("selected", SURFACE2)],
              foreground=[("selected", TEXT)])

        self._tab_annotate = ttk.Frame(nb)
        self._tab_augment  = ttk.Frame(nb)
        self._tab_stats    = ttk.Frame(nb)
        self._tab_full     = ttk.Frame(nb)
        self._tab_log      = ttk.Frame(nb)

        nb.add(self._tab_annotate, text="  Annotate  ")
        nb.add(self._tab_augment,  text="  Augment  ")
        nb.add(self._tab_stats,    text="  Statistics  ")
        nb.add(self._tab_full,     text="  Full pipeline  ")
        nb.add(self._tab_log,      text="  Log  ")

        self._build_tab_annotate()
        self._build_tab_augment()
        self._build_tab_stats()
        self._build_tab_full()
        self._build_tab_log()

        # Shared terminal at the bottom
        ttk.Label(parent, text="Terminal output",
                  style="Muted2.TLabel").pack(anchor="w", pady=(8, 2))
        self._terminal = scrolledtext.ScrolledText(
            parent, height=10, bg="#0d0d1a", fg=TEXT,
            font=("Consolas", 9), relief="flat", bd=0,
            state="disabled"
        )
        self._terminal.pack(fill="both", expand=False)
        self._terminal.tag_config("cmd", foreground=ACCENT,
                                  font=("Consolas", 9, "bold"))
        self._terminal.tag_config("ok",  foreground=SUCCESS)
        self._terminal.tag_config("err", foreground=DANGER)

    # ── Tab: Annotate ─────────────────────────────────────────────────────
    def _build_tab_annotate(self):
        p = self._tab_annotate
        self._section(p, "Annotation tool",
                      "Opens the Tkinter YOLO labelling tool.\n"
                      "Images are assigned automatically based on lock state.")

        opts = ttk.Frame(p, style="Card.TFrame", padding=12)
        opts.pack(fill="x", padx=16, pady=8)

        self.var_reannotate = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Re-annotate already-done images",
                        variable=self.var_reannotate).pack(anchor="w")

        ttk.Button(p, text="Launch annotation tool  →",
                   style="Accent.TButton",
                   command=self._run_annotate).pack(padx=16, pady=8,
                                                    anchor="w")

        self._section(p, "How it works",
                      "• On start: git pull → stale locks expire → tool\n"
                      "  jumps to your first unclaimed image automatically.\n"
                      "• Press  [n]  to save + advance (one git push).\n"
                      "• Press  [s]  to save without advancing.\n"
                      "• Press  [q]  to release lock and quit.")

    def _run_annotate(self):
        dev = self.var_dev.get().strip()
        if not dev:
            messagebox.showerror("Missing", "Enter your developer name first.")
            return
        cmd = [sys.executable, str(SCRIPT), "annotate",
               "--images", self.var_images.get(),
               "--labels", self.var_labels.get(),
               "--dev",    dev]
        if self.var_reannotate.get():
            cmd.append("--reannotate")
        _run_cmd(cmd, self._terminal, on_done=self._refresh_stats)

    # ── Tab: Augment ──────────────────────────────────────────────────────
    def _build_tab_augment(self):
        p = self._tab_augment
        self._section(p, "Augmentation",
                      "Applies bbox-aware transforms to all annotated images\n"
                      "that haven't been augmented yet.")

        opts = ttk.Frame(p, style="Card.TFrame", padding=12)
        opts.pack(fill="x", padx=16, pady=8)
        opts.columnconfigure(1, weight=1)

        # n augmentations
        ttk.Label(opts, text="Augmentations / image:", style="Muted.TLabel",
                  ).grid(row=0, column=0, sticky="w", pady=4)
        self.var_n = tk.StringVar(value="4")
        ttk.Entry(opts, textvariable=self.var_n, width=6).grid(
            row=0, column=1, sticky="w", padx=8)

        # image size
        ttk.Label(opts, text="Output size (px):", style="Muted.TLabel",
                  ).grid(row=1, column=0, sticky="w", pady=4)
        self.var_size = tk.StringVar(value="640")
        ttk.Combobox(opts, textvariable=self.var_size,
                     values=["416", "512", "640", "768", "1024"],
                     width=8, state="readonly").grid(
            row=1, column=1, sticky="w", padx=8)

        # toggles
        self.var_mosaic = tk.BooleanVar(value=True)
        self.var_mixup  = tk.BooleanVar(value=True)
        self.var_force  = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Mosaic augmentation",
                        variable=self.var_mosaic).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(opts, text="MixUp augmentation",
                        variable=self.var_mixup).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(opts, text="Force re-augment (skip idempotency check)",
                        variable=self.var_force).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Button(p, text="Run augmentation  →",
                   style="Accent.TButton",
                   command=self._run_augment).pack(padx=16, pady=8,
                                                   anchor="w")

    def _run_augment(self):
        try:
            n = int(self.var_n.get())
        except ValueError:
            messagebox.showerror("Invalid", "Augmentations/image must be integer.")
            return
        cmd = [sys.executable, str(SCRIPT), "augment",
               "--images",     self.var_images.get(),
               "--labels",     self.var_labels.get(),
               "--out-images", self.var_out_img.get(),
               "--out-labels", self.var_out_lbl.get(),
               "--n",          str(n),
               "--size",       self.var_size.get()]
        if not self.var_mosaic.get():
            cmd.append("--no-mosaic")
        if not self.var_mixup.get():
            cmd.append("--no-mixup")
        if self.var_force.get():
            cmd.append("--force")
        _run_cmd(cmd, self._terminal, on_done=self._refresh_stats)

    # ── Tab: Statistics ───────────────────────────────────────────────────
    def _build_tab_stats(self):
        p = self._tab_stats
        self._section(p, "Annotation statistics",
                      "Counts boxes per class → saves YAML + Markdown report.")

        opts = ttk.Frame(p, style="Card.TFrame", padding=12)
        opts.pack(fill="x", padx=16, pady=8)
        opts.columnconfigure(1, weight=1)

        ttk.Label(opts, text="YAML output:", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", pady=4)
        self.var_yaml = tk.StringVar(value="annotation_stats.yaml")
        ttk.Entry(opts, textvariable=self.var_yaml).grid(
            row=0, column=1, sticky="ew", padx=8)

        ttk.Label(opts, text="Report output:", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=4)
        self.var_report = tk.StringVar(value="annotation_report.md")
        ttk.Entry(opts, textvariable=self.var_report).grid(
            row=1, column=1, sticky="ew", padx=8)

        self.var_publish = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Publish to GitHub PR via CML",
                        variable=self.var_publish).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=4)

        btn_row = ttk.Frame(p)
        btn_row.pack(padx=16, pady=8, anchor="w")
        ttk.Button(btn_row, text="Compute statistics  →",
                   style="Accent.TButton",
                   command=self._run_stats).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Open report",
                   command=self._open_report).pack(side="left")

    def _run_stats(self):
        cmd = [sys.executable, str(SCRIPT), "stats",
               "--labels", self.var_labels.get(),
               "--yaml",   self.var_yaml.get(),
               "--report", self.var_report.get()]
        if self.var_publish.get():
            cmd.append("--publish")
        _run_cmd(cmd, self._terminal)

    def _open_report(self):
        rp = Path(self.var_report.get())
        if not rp.exists():
            messagebox.showinfo("Not found",
                                "Run statistics first to generate the report.")
            return
        import os
        os.startfile(str(rp)) if sys.platform == "win32" else \
            subprocess.Popen(["xdg-open", str(rp)])

    # ── Tab: Full pipeline ────────────────────────────────────────────────
    def _build_tab_full(self):
        p = self._tab_full
        self._section(p, "Full pipeline",
                      "Runs statistics + augmentation in one step.")

        opts = ttk.Frame(p, style="Card.TFrame", padding=12)
        opts.pack(fill="x", padx=16, pady=8)
        opts.columnconfigure(1, weight=1)

        ttk.Label(opts, text="Augmentations / image:", style="Muted.TLabel",
                  ).grid(row=0, column=0, sticky="w", pady=4)
        self.var_fn = tk.StringVar(value="4")
        ttk.Entry(opts, textvariable=self.var_fn, width=6).grid(
            row=0, column=1, sticky="w", padx=8)

        ttk.Label(opts, text="Output size:", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=4)
        self.var_fsize = tk.StringVar(value="640")
        ttk.Combobox(opts, textvariable=self.var_fsize,
                     values=["416", "512", "640", "768", "1024"],
                     width=8, state="readonly").grid(
            row=1, column=1, sticky="w", padx=8)

        self.var_fpublish = tk.BooleanVar(value=False)
        self.var_fforce   = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Publish report to GitHub PR",
                        variable=self.var_fpublish).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(opts, text="Force re-augment",
                        variable=self.var_fforce).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Button(p, text="Run full pipeline  →",
                   style="Accent.TButton",
                   command=self._run_full).pack(padx=16, pady=8, anchor="w")

    def _run_full(self):
        cmd = [sys.executable, str(SCRIPT), "full",
               "--images",     self.var_images.get(),
               "--labels",     self.var_labels.get(),
               "--out-images", self.var_out_img.get(),
               "--out-labels", self.var_out_lbl.get(),
               "--n",          self.var_fn.get(),
               "--size",       self.var_fsize.get()]
        if self.var_fpublish.get():
            cmd.append("--publish")
        if self.var_fforce.get():
            cmd.append("--force")
        _run_cmd(cmd, self._terminal, on_done=self._refresh_stats)

    # ── Tab: Log viewer ───────────────────────────────────────────────────
    def _build_tab_log(self):
        p = self._tab_log
        self._section(p, "Processed log",
                      "Live view of processed_log.json — who annotated what,\n"
                      "which images are augmented, and active locks.")

        btn_row = ttk.Frame(p)
        btn_row.pack(padx=16, pady=(0, 6), anchor="w")
        ttk.Button(btn_row, text="⟳  Refresh log",
                   command=self._refresh_log_tab).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Run  pipeline.py log",
                   command=lambda: _run_cmd(
                       [sys.executable, str(SCRIPT), "log"],
                       self._terminal)).pack(side="left")

        self._log_text = scrolledtext.ScrolledText(
            p, bg=SURFACE, fg=TEXT, font=("Consolas", 9),
            relief="flat", bd=0
        )
        self._log_text.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self._log_text.tag_config("section", foreground=ACCENT,
                                  font=("Consolas", 9, "bold"))
        self._log_text.tag_config("done",    foreground=SUCCESS)
        self._log_text.tag_config("lock",    foreground=WARNING)
        self._log_text.tag_config("key",     foreground=MUTED)
        self._refresh_log_tab()

    def _refresh_log_tab(self):
        log = _load_log()
        t   = self._log_text
        t.config(state="normal")
        t.delete("1.0", "end")

        def row(k, v, tag=""):
            t.insert("end", f"  {k:<28}", "key")
            t.insert("end", f"{v}\n", tag)

        t.insert("end", "── ANNOTATED ──────────────────────────────\n", "section")
        if log["annotated"]:
            for stem, info in sorted(log["annotated"].items()):
                row(stem, f"{info.get('boxes',0)} boxes  by {info.get('annotated_by','?')}", "done")
        else:
            t.insert("end", "  (none)\n")

        t.insert("end", "\n── AUGMENTED ──────────────────────────────\n", "section")
        if log["augmented"]:
            for stem, info in sorted(log["augmented"].items()):
                row(stem, f"{info.get('num_augmented',0)} variants  {info.get('timestamp','')}", "done")
        else:
            t.insert("end", "  (none)\n")

        t.insert("end", "\n── ACTIVE LOCKS ────────────────────────────\n", "section")
        active = {s: e for s, e in log.get("locks", {}).items()
                  if not _is_expired(e)}
        if active:
            for stem, info in active.items():
                row(stem, f"🔒 {info['locked_by']}  expires {info['expires_at']}", "lock")
        else:
            t.insert("end", "  (no active locks)\n")

        t.config(state="disabled")

    # ── Stats refresh ─────────────────────────────────────────────────────
    def _refresh_stats(self):
        log    = _load_log()
        ann    = log.get("annotated", {})
        aug    = log.get("augmented", {})
        locks  = {s: e for s, e in log.get("locks", {}).items()
                  if not _is_expired(e)}

        self._stat_ann.set(str(len(ann)))
        self._stat_aug.set(str(len(aug)))
        self._stat_locks.set(str(len(locks)))

        self._locks_box.config(state="normal")
        self._locks_box.delete("1.0", "end")
        if locks:
            for stem, info in locks.items():
                self._locks_box.insert(
                    "end",
                    f"🔒 {stem:<30} → {info['locked_by']}\n"
                )
        else:
            self._locks_box.insert("end", "No active locks.\n")
        self._locks_box.config(state="disabled")

    def _schedule_refresh(self):
        self._refresh_stats()
        self.after(5000, self._schedule_refresh)

    # ── Section helper ────────────────────────────────────────────────────
    def _section(self, parent, title: str, body: str = ""):
        f = ttk.Frame(parent)
        f.pack(fill="x", padx=16, pady=(12, 4))
        ttk.Label(f, text=title, style="H2b.TLabel").pack(anchor="w")
        if body:
            ttk.Label(f, text=body, style="Muted2.TLabel",
                      justify="left").pack(anchor="w", pady=(2, 0))
        ttk.Separator(parent).pack(fill="x", padx=16, pady=(4, 4))


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = PipelineGUI()
    app.mainloop()
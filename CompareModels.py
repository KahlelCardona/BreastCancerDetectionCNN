import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pathlib import Path

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
PLOT_DIRS = {
    "EfficientNet":  Path("plots_efficientnet"),
    "ResNet":        Path("plots_resnet"),
    "Distillation":  Path("plots_distill_kfold"),
}

METRIC_FILES = {
    "Loss":     "mean_val_loss.png",
    "Accuracy": "mean_val_acc.png",
    "F1 Score": "mean_val_f1.png",
}

# ─────────────────────────────────────────────
#  Palette  (dark clinical)
# ─────────────────────────────────────────────
BG          = "#0d1117"   # near-black background
PANEL       = "#161b22"   # card surface
BORDER      = "#21262d"   # subtle border
ACCENT      = "#58a6ff"   # electric blue — primary action
ACCENT2     = "#3fb950"   # green — success / F1
ACCENT3     = "#f78166"   # coral — loss / warning
TEXT_HI     = "#e6edf3"   # primary text
TEXT_MID    = "#8b949e"   # secondary text
TEXT_LO     = "#484f58"   # disabled / hint
SEL_BG      = "#1f6feb"   # selected state bg
HOVER       = "#1c2128"   # hover bg

FONT_TITLE  = ("Courier New", 13, "bold")
FONT_LABEL  = ("Courier New", 9)
FONT_BTN    = ("Courier New", 9, "bold")
FONT_STATUS = ("Courier New", 8)

# ─────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────
def get_image_path(model, metric, fold):
    base_dir = PLOT_DIRS.get(model)
    if base_dir is None or not base_dir.exists():
        return None
    if fold == "mean":
        fname = METRIC_FILES.get(metric)
        return None if fname is None else base_dir / fname
    return base_dir / f"fold_{fold}_metrics.png"


def load_and_display(canvas, ax, image_path):
    ax.clear()
    ax.set_facecolor(PANEL)
    if not image_path or not image_path.exists():
        ax.text(0.5, 0.5, "— no plot found —",
                ha="center", va="center",
                color=TEXT_LO, fontsize=11, fontfamily="Courier New")
        ax.axis("off")
    else:
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.axis("off")
    canvas.draw()


# ─────────────────────────────────────────────
#  Custom segmented-button row
# ─────────────────────────────────────────────
class SegmentedRow(tk.Frame):
    """A row of toggle buttons where exactly one is selected at a time."""

    def __init__(self, parent, options, callback, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self.var = tk.StringVar(value=options[0])
        self.buttons = {}
        self.callback = callback

        for i, opt in enumerate(options):
            # rounded illusion via tight padding
            btn = tk.Label(
                self, text=opt,
                font=FONT_BTN,
                padx=14, pady=6,
                cursor="hand2",
                relief="flat",
            )
            btn.grid(row=0, column=i, padx=1, pady=0)
            btn.bind("<Button-1>", lambda e, o=opt: self._select(o))
            btn.bind("<Enter>",    lambda e, b=btn, o=opt: self._hover(b, o))
            btn.bind("<Leave>",    lambda e, b=btn, o=opt: self._unhover(b, o))
            self.buttons[opt] = btn

        self._refresh()

    def _select(self, opt):
        self.var.set(opt)
        self._refresh()
        self.callback()

    def _hover(self, btn, opt):
        if self.var.get() != opt:
            btn.configure(bg=HOVER, fg=TEXT_HI)

    def _unhover(self, btn, opt):
        self._refresh_one(btn, opt)

    def _refresh(self):
        for opt, btn in self.buttons.items():
            self._refresh_one(btn, opt)

    def _refresh_one(self, btn, opt):
        if self.var.get() == opt:
            btn.configure(bg=ACCENT, fg="#ffffff")
        else:
            btn.configure(bg=PANEL, fg=TEXT_MID)

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(value)
        self._refresh()


# ─────────────────────────────────────────────
#  Main UI
# ─────────────────────────────────────────────
class ModelComparisonUI:
    def __init__(self, root):
        self.root = root
        root.title("Mammogram Model Comparison")
        root.configure(bg=BG)
        root.geometry("1020x720")
        root.resizable(True, True)

        self._build_header()
        self._build_controls()
        self._build_canvas()
        self._build_statusbar()

        self.update_view()

    # ── Header ──────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=PANEL, height=52)
        hdr.pack(fill="x", side="top")

        # left accent bar
        tk.Frame(hdr, bg=ACCENT, width=4).pack(side="left", fill="y")

        tk.Label(
            hdr, text="MAMMOGRAM  CLASSIFICATION",
            font=FONT_TITLE, bg=PANEL, fg=TEXT_HI,
            padx=18, pady=14,
        ).pack(side="left")

        tk.Label(
            hdr, text="model comparison dashboard",
            font=FONT_STATUS, bg=PANEL, fg=TEXT_LO,
            pady=14,
        ).pack(side="left")

        # right tag
        tag = tk.Label(
            hdr, text="ResNet · EfficientNet · Distillation",
            font=FONT_STATUS, bg=PANEL, fg=TEXT_MID,
            padx=18, pady=14,
        )
        tag.pack(side="right")

        # separator line
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

    # ── Control strip ───────────────────────
    def _build_controls(self):
        strip = tk.Frame(self.root, bg=BG, pady=14)
        strip.pack(fill="x", padx=24)

        def section(label_text):
            col = tk.Frame(strip, bg=BG)
            col.pack(side="left", padx=(0, 32))
            tk.Label(col, text=label_text,
                     font=FONT_STATUS, bg=BG, fg=TEXT_LO).pack(anchor="w", pady=(0, 4))
            return col

        # MODEL
        mcol = section("MODEL")
        self.model_seg = SegmentedRow(
            mcol,
            list(PLOT_DIRS.keys()),
            self.update_view,
        )
        self.model_seg.pack()

        # METRIC
        ecol = section("METRIC")
        self.metric_seg = SegmentedRow(
            ecol,
            list(METRIC_FILES.keys()),
            self.update_view,
        )
        self.metric_seg.pack()

        # FOLD
        fcol = section("FOLD")
        fold_options = ["mean"] + [f"Fold {i}" for i in range(1, 6)]
        self.fold_seg = SegmentedRow(
            fcol,
            fold_options,
            self.update_view,
        )
        self.fold_seg.pack()

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

    # ── Plot canvas ─────────────────────────
    def _build_canvas(self):
        wrapper = tk.Frame(self.root, bg=BG)
        wrapper.pack(fill="both", expand=True, padx=20, pady=16)

        self.figure = Figure(facecolor=BG)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(PANEL)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.canvas = FigureCanvasTkAgg(self.figure, master=wrapper)
        widget = self.canvas.get_tk_widget()
        widget.configure(bg=BG, highlightthickness=0)
        widget.pack(fill="both", expand=True)

    # ── Status bar ──────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=PANEL, height=28)
        bar.pack(fill="x", side="bottom")
        tk.Frame(bar, bg=BORDER, height=1).pack(fill="x", side="top")

        self.status_var = tk.StringVar(value="ready")
        tk.Label(
            bar, textvariable=self.status_var,
            font=FONT_STATUS, bg=PANEL, fg=TEXT_MID,
            padx=16, pady=6,
        ).pack(side="left")

        # right side: path hint
        self.path_var = tk.StringVar(value="")
        tk.Label(
            bar, textvariable=self.path_var,
            font=FONT_STATUS, bg=PANEL, fg=TEXT_LO,
            padx=16, pady=6,
        ).pack(side="right")

    # ── Update ──────────────────────────────
    def update_view(self, *_):
        model    = self.model_seg.get()
        metric   = self.metric_seg.get()
        fold_str = self.fold_seg.get()

        fold = "mean" if fold_str == "mean" else int(fold_str.split()[1])

        img_path = get_image_path(model, metric, fold)

        load_and_display(self.canvas, self.ax, img_path)

        if img_path is None:
            self.status_var.set(f"✗  no directory found for  {model}")
            self.path_var.set("")
        elif not img_path.exists():
            self.status_var.set(f"✗  file missing")
            self.path_var.set(str(img_path))
        else:
            self.status_var.set(
                f"●  {model}  ·  {metric}  ·  {fold_str}"
            )
            self.path_var.set(str(img_path))


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = ModelComparisonUI(root)
    root.mainloop()
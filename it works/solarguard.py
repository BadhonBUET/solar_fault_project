import os
import csv
import copy
import traceback
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ── Import the core pipeline (v3) ────────────────────────────────────────────
# Uses the new pipeline file under its default name; if you saved it as final1.py
# instead, that is picked up too.
try:
    import onlyfault as pv
except ImportError:
    import final1 as pv

if not hasattr(pv, "analyse"):
    raise ImportError("The imported pipeline is the OLD version (no analyse()). "
                      "Save the v3 code as pv_fault_detection.py (or overwrite final1.py) "
                      "in the same folder as this GUI.")


# ==============================================================================
# Pipeline -> GUI helpers  (pure functions: no Tk in here, easy to test)
# ==============================================================================

STAGES = (
    "1. Raw Thermal Input",
    "2. FFT Spectrum (Tilt Check)",
    "3. LTI Variance Map",
    "4. Panel Evidence (LTI + DWT)",
    "5. Panel ROI",
    "6. Thermal Delta Map (ROI)",
    "7. Hard-Wavelet Map + Faults",
    "8. Final Annotated Output",
)

SOURCE_LABEL = {"thermal": "Thermal", "abrupt": "Abrupt", "thermal+abrupt": "Both"}


def build_stage_images(R):
    """One entry per stage:  {img, cmap, kw (imshow kwargs), note (text under the image)}."""
    t, ga = R.tilt, R.tilt.gray_aligned
    S = STAGES
    st = {}

    st[S[0]] = dict(img=R.gray_raw, cmap="gray")

    st[S[1]] = dict(img=t.fft_mag, cmap="inferno",
                    note=f"FFT {t.fft_angle:.1f}°   |   Projection scan {t.angle:.1f}°   |   "
                         f"{'agree ✓' if t.agree else 'DISAGREE ✗ - check image quality'}")

    st[S[2]] = dict(img=R.lti.std_map, cmap="viridis",
                    note="dark = smooth panel surface (log-variance, automatic Otsu cut)")

    st[S[3]] = dict(img=R.roi_res.evidence,
                    note="grey = LTI panel area   |   cyan / orange = DWT horizontal / vertical borders")

    # 5. ROI on the aligned image: green = final ROI, cyan = verified interior used for detection
    ov = cv2.cvtColor(ga, cv2.COLOR_GRAY2RGB)
    cnts, _ = cv2.findContours(R.roi_res.roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2)
    for (rx, ry, rw, rh) in R.roi_res.core_rects:
        cv2.rectangle(ov, (rx, ry), (rx + rw - 1, ry + rh - 1), (0, 200, 255), 1)
    st[S[4]] = dict(img=ov, note="green = panel ROI (closing → inside check → DWT border → 1 px pad)   |   "
                                 "cyan = verified interior used for detection")

    th = R.th
    if th is None:                                              # no panel found: show empty maps
        blank = np.zeros((*ga.shape, 3), np.uint8)
        st[S[5]] = dict(img=blank, note="No panel ROI found")
        st[S[6]] = dict(img=blank, note="No panel ROI found")
    else:
        # 6. thermal delta (ROI only)
        show = np.maximum(np.where(th.inside, th.delta, 0.0), 0.0)
        vmax = max(float(np.percentile(show[th.inside], 99.8)), 3.0 * th.T_hi)
        st[S[5]] = dict(img=show, cmap="magma", kw=dict(vmin=0, vmax=vmax),
                        note=f"noise σ = {th.sigma:.1f}   |   seed {th.T_hi:.0f} / grow {th.T_lo:.0f} grey levels")

        # 7. red = thermal excess, green = abrupt brightness increase, boxes = accepted faults
        comp = np.stack([np.clip(th.enhanced / (2.0 * th.T_hi), 0, 1),
                         np.clip(th.abrupt / (4.0 * th.T_ab), 0, 1),
                         np.zeros_like(th.enhanced)], axis=-1)
        comp = (comp * 255).astype(np.uint8)
        for f in R.faults:
            rgb = (int(f["color"][2]), int(f["color"][1]), int(f["color"][0]))       # BGR -> RGB
            cv2.polylines(comp, [f["rect_pts"].astype(np.int32).reshape(-1, 1, 2)], True, rgb, 1)
        st[S[6]] = dict(img=comp, note=f"hard-wavelet tau = {th.tau:.0f}   |   red = thermal excess   "
                                       f"green = abrupt brightness increase   |   boxes = accepted faults")

    c = R.counts
    st[S[7]] = dict(img=cv2.cvtColor(R.annotated, cv2.COLOR_BGR2RGB),
                    note=f"Hotspots {c['Hotspot']}   |   Line faults {c['Line Fault']}   |   "
                         f"Partial shading {c['Partial Shading']}   (label = brightness % and FSI)")
    return st


def meta_summary(R):
    """Text for the 'Scan Diagnostics' box (kept under ~38 characters per line)."""
    t, c = R.tilt, R.counts
    lines = [f"Tilt: {t.angle:.1f}°  (FFT {'ok' if t.agree else 'DISAGREES'})",
             f"ROI:  {len(R.roi_res.rects)} row(s), {100 * R.roi_res.coverage:.0f}% of frame",
             f"Cell: {R.geom.cell_w:.0f} x {R.geom.cell_h:.0f} px"]
    if R.roi_mult != 1.0:
        lines.append(f"ROI cutoff auto-adjusted x{R.roi_mult:g}")
    if R.th is None:
        lines.append("No panel ROI found")
    else:
        th = R.th
        lines += [f"Noise σ: {th.sigma:.1f}   Range: {th.dyn:.0f}",
                  f"Thr seed/grow/abrupt: {th.T_hi:.0f}/{th.T_lo:.0f}/{th.T_ab:.0f}",
                  f"Max panel bright: {float(R.tilt.gray_aligned[th.inside].max()):.0f}"]
    lines += [f"Total Faults:   {sum(c.values())}",
              f" - Hotspots:    {c['Hotspot']}",
              f" - Line Faults: {c['Line Fault']}",
              f" - Partial Shading: {c['Partial Shading']}"]
    return "\n".join(lines)


def fault_rows(R):
    """Rows for the fault table (already ranked by FSI in the pipeline)."""
    return [(f["type"], f["brightness_pct"], f["fsi"], SOURCE_LABEL.get(f["source"], f["source"]))
            for f in R.faults]


def csv_rows(R):
    """Header + one row per fault.  Position is given in ORIGINAL (un-rotated) image pixels."""
    inv = cv2.invertAffineTransform(R.tilt.rot_mat)
    header = ["Fault Type", "FSI Score", "Brightness (%)", "Heat (0-1)", "Aspect Ratio", "Length x Width (px)",
              "Orientation", "Detected By", "IIR Confirmed", "Center X (original)", "Center Y (original)",
              "Bounding Box (aligned frame x, y, w, h)"]
    rows = [header]
    for f in R.faults:
        cx, cy = f["rect_pts"].mean(axis=0)
        ox, oy = inv @ np.array([cx, cy, 1.0])
        iir = "n/a" if f["iir_confirmed"] is None else ("yes" if f["iir_confirmed"] else "no")
        rows.append([f["type"], f["fsi"], f["brightness_pct"], f"{f['heat']:.2f}", f"{f['aspect']:.1f}",
                     f"{f['L']:.0f} x {f['W']:.0f}", "horizontal" if f["horizontal"] else "vertical",
                     SOURCE_LABEL.get(f["source"], f["source"]), iir, int(round(ox)), int(round(oy)), f["box"]])
    return rows


# ==============================================================================
# GUI
# ==============================================================================

class ThermalPVDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("SolarGuard: Thermal PV Diagnostics Dashboard")
        self.root.geometry("1200x900")
        self.root.minsize(1100, 820)

        # Color Palette
        self.bg_color = "#161623"
        self.panel_color = "#22223b"
        self.text_color = "#f2e9e4"
        self.accent_color = "#4a4e69"
        self.button_hover = "#9a8c98"
        self.root.configure(bg=self.bg_color)

        self.stages = STAGES
        self.image_data = {}
        self.current_stage = tk.StringVar(value=self.stages[0])
        self.current_faults = []

        # pipeline state (the tilt is computed once per image; sliders only re-run what follows it)
        self.base_cfg = pv.Config()
        self.gray_raw = None
        self.img_bgr = None
        self.tilt = None
        self.result = None
        self._after_id = None
        self.sliders = []                       # (config group, attribute, tk.Scale)

        self.setup_styles()
        self.setup_ui()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Improved Combobox Styling
        style.configure("TCombobox",
                        fieldbackground=self.panel_color,
                        background=self.accent_color,
                        foreground="white",
                        arrowcolor="white",
                        padding=5)

        # Treeview Styling
        style.configure("Treeview", background=self.panel_color, foreground=self.text_color,
                        fieldbackground=self.panel_color, borderwidth=0)
        style.configure("Treeview.Heading", background=self.accent_color, foreground="white",
                        font=("Arial", 10, "bold"))
        style.map("Treeview", background=[('selected', self.button_hover)])

    def setup_ui(self):
        # ── Left Sidebar (Controls & Data) ──
        sidebar = tk.Frame(self.root, bg=self.panel_color, width=350, padx=20, pady=15)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Bottom block first, so Export / status stay visible even on a small screen
        bottom = tk.Frame(sidebar, bg=self.panel_color)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="")
        tk.Label(bottom, textvariable=self.status_var, font=("Arial", 9, "italic"),
                 bg=self.panel_color, fg="#c9ada7").pack(anchor="w", pady=(0, 4))
        self.export_btn = tk.Button(bottom, text="💾 Export Fault CSV", command=self.export_csv,
                                    bg=self.accent_color, fg="white", font=("Arial", 10, "bold"),
                                    relief="flat", state=tk.DISABLED)
        self.export_btn.pack(fill=tk.X)

        # Header
        tk.Label(sidebar, text="PV Inspection Panel", font=("Arial", 16, "bold"),
                 bg=self.panel_color, fg="white").pack(pady=(0, 12))

        # Upload Button
        upload_btn = tk.Button(sidebar, text="📁 Upload Thermal Image", command=self.process_image,
                               bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), relief="flat", pady=8)
        upload_btn.pack(fill=tk.X, pady=(0, 12))

        # ── Stage Navigator ──
        tk.Label(sidebar, text="Pipeline Stage:", font=("Arial", 10, "bold"),
                 bg=self.panel_color, fg="#c9ada7").pack(anchor="w", pady=(0, 4))

        nav_frame = tk.Frame(sidebar, bg=self.panel_color)
        nav_frame.pack(fill=tk.X, pady=(0, 12))

        self.prev_btn = tk.Button(nav_frame, text="◀", command=self.prev_stage, bg=self.accent_color, fg="white",
                                  font=("Arial", 10, "bold"), relief="flat", state=tk.DISABLED, width=3)
        self.prev_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.stage_dropdown = ttk.Combobox(nav_frame, textvariable=self.current_stage, state="readonly",
                                           values=self.stages, font=("Arial", 10))
        self.stage_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.stage_dropdown.bind("<<ComboboxSelected>>", self.update_display)

        self.next_btn = tk.Button(nav_frame, text="▶", command=self.next_stage, bg=self.accent_color, fg="white",
                                  font=("Arial", 10, "bold"), relief="flat", state=tk.DISABLED, width=3)
        self.next_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # ── Threshold sliders (same two knobs as the pipeline's own tuner) ──
        tk.Label(sidebar, text="Detection Tuning:", font=("Arial", 12, "bold"),
                 bg=self.panel_color, fg="white").pack(anchor="w", pady=(0, 2))
        for label, grp, attr, lo, hi in pv.TUNER_SLIDERS:
            sc = tk.Scale(sidebar, from_=lo, to=hi, resolution=0.05, orient=tk.HORIZONTAL, label=label,
                          command=self.on_slider, bg=self.panel_color, fg="white", troughcolor=self.bg_color,
                          activebackground=self.button_hover, highlightthickness=0, bd=0, length=300,
                          font=("Arial", 9))
            sc.set(float(getattr(getattr(self.base_cfg, grp), attr)))
            sc.pack(fill=tk.X)
            self.sliders.append((grp, attr, sc))
        tk.Label(sidebar, text="Right = catch weaker anomalies / larger ROI.  Left = stricter.",
                 font=("Arial", 8), bg=self.panel_color, fg="#9a8c98", justify=tk.LEFT).pack(anchor="w", pady=(0, 2))
        tk.Button(sidebar, text="Reset sliders", command=self.reset_sliders, bg=self.accent_color, fg="white",
                  font=("Arial", 8, "bold"), relief="flat").pack(anchor="e", pady=(0, 8))

        # ── Metadata Section ──
        tk.Label(sidebar, text="Scan Diagnostics:", font=("Arial", 12, "bold"),
                 bg=self.panel_color, fg="white").pack(anchor="w", pady=(0, 3))
        self.meta_text = tk.StringVar(value="Waiting for image...")
        tk.Label(sidebar, textvariable=self.meta_text, justify=tk.LEFT, anchor="nw", height=12,
                 font=("Consolas", 9), bg=self.panel_color, fg="#9a8c98").pack(anchor="w", fill=tk.X, pady=(0, 8))

        # ── Fault Table ──
        tk.Label(sidebar, text="Detected Faults (Ranked by FSI):", font=("Arial", 11, "bold"),
                 bg=self.panel_color, fg="white").pack(anchor="w", pady=(0, 3))

        columns = ("Type", "Bright", "FSI", "Source")
        self.tree = ttk.Treeview(sidebar, columns=columns, show="headings", height=6)
        self.tree.heading("Type", text="Fault Type")
        self.tree.heading("Bright", text="Bright %")
        self.tree.heading("FSI", text="FSI")
        self.tree.heading("Source", text="Found by")
        self.tree.column("Type", width=105)
        self.tree.column("Bright", width=60, anchor=tk.CENTER)
        self.tree.column("FSI", width=50, anchor=tk.CENTER)
        self.tree.column("Source", width=75, anchor=tk.CENTER)
        self.tree.pack(fill=tk.X, pady=(0, 8))

        # ── Right Main Area (Visualizer) ──
        main_area = tk.Frame(self.root, bg=self.bg_color)
        main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor(self.bg_color)
        self.ax.set_facecolor(self.bg_color)
        self.ax.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=main_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=(20, 0))

        # Interactive Toolbar
        toolbar_frame = tk.Frame(main_area, bg=self.bg_color)
        toolbar_frame.pack(fill=tk.X, padx=20, pady=10)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.toolbar.config(background=self.bg_color)
        for button in self.toolbar.winfo_children():
            button.config(background=self.bg_color)

    # ── Navigation Logic ──
    def prev_stage(self):
        current_idx = self.stages.index(self.current_stage.get())
        if current_idx > 0:
            self.current_stage.set(self.stages[current_idx - 1])
            self.update_display()

    def next_stage(self):
        current_idx = self.stages.index(self.current_stage.get())
        if current_idx < len(self.stages) - 1:
            self.current_stage.set(self.stages[current_idx + 1])
            self.update_display()

    # ── Sliders ──
    def on_slider(self, _value=None):
        """Called continuously while a slider moves: wait until it rests, then re-run the analysis."""
        if self.tilt is None:                       # no image loaded yet
            return
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(250, self.reanalyse)

    def reset_sliders(self):
        defaults = pv.Config()
        for grp, attr, sc in self.sliders:
            sc.set(float(getattr(getattr(defaults, grp), attr)))
        self.on_slider()

    def current_cfg(self):
        cfg = copy.deepcopy(self.base_cfg)
        for grp, attr, sc in self.sliders:
            setattr(getattr(cfg, grp), attr, float(sc.get()))
        return cfg

    # ── Processing ──
    def process_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Local Test Data",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not file_path:
            return

        try:
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                raise ValueError("Could not open the selected image.")
            self.status_var.set("Aligning tilt...")
            self.root.update_idletasks()
            self.img_bgr = img_bgr
            self.gray_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # tilt search + FFT check: once per image (the sliders do not change it)
            self.tilt = pv.detect_and_align_tilt(self.gray_raw, self.base_cfg.tilt)
            self.reanalyse(reset_stage=True)
        except Exception as e:
            traceback.print_exc()
            self.status_var.set("")
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")

    def reanalyse(self, reset_stage=False):
        """Run everything after the tilt step with the current slider values and refresh the GUI."""
        self._after_id = None
        if self.tilt is None:
            return
        try:
            self.status_var.set("Analysing...")
            self.root.update_idletasks()

            R = pv.analyse(self.gray_raw, self.img_bgr, self.tilt, self.current_cfg())
            pv.print_summary(R)                          # derived thresholds also go to the console
            self.result = R
            self.current_faults = R.faults

            for item in self.tree.get_children():
                self.tree.delete(item)
            for row in fault_rows(R):
                self.tree.insert("", tk.END, values=row)

            self.meta_text.set(meta_summary(R))
            self.export_btn.config(state=tk.NORMAL)
            self.image_data = build_stage_images(R)

            if reset_stage:
                self.current_stage.set(self.stages[0])
            self.update_display()
            self.status_var.set("")
        except Exception as e:
            traceback.print_exc()
            self.status_var.set("")
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")

    def update_display(self, event=None):
        if not self.image_data:
            return

        stage_key = self.current_stage.get()
        current_idx = self.stages.index(stage_key)

        # Dynamically enable/disable navigation buttons based on current stage
        self.prev_btn.config(state=tk.NORMAL if current_idx > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if current_idx < len(self.stages) - 1 else tk.DISABLED)

        entry = self.image_data.get(stage_key)

        self.ax.clear()
        self.ax.axis('off')

        if entry is not None:
            self.ax.imshow(entry["img"], cmap=entry.get("cmap"), **entry.get("kw", {}))
            self.ax.set_title(stage_key, color="white", fontsize=14, fontweight="bold", pad=15)
            if entry.get("note"):
                self.ax.text(0.5, -0.01, entry["note"], transform=self.ax.transAxes, ha="center", va="top",
                             color="#c9ada7", fontsize=9)

        self.canvas.draw()

    def export_csv(self):
        if not self.current_faults:
            messagebox.showinfo("Export", "No faults detected to export.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Fault Report"
        )

        if save_path:
            with open(save_path, mode='w', newline='', encoding='utf-8') as file:
                csv.writer(file).writerows(csv_rows(self.result))
            messagebox.showinfo("Success", f"Report saved to:\n{os.path.basename(save_path)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalPVDashboard(root)
    root.mainloop()
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import csv
import os

# Import the core processing blocks from your existing project
import final1

class ThermalPVDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("SolarGuard: Thermal PV Diagnostics Dashboard")
        self.root.geometry("1100x800")
        
        # Color Palette
        self.bg_color = "#161623"
        self.panel_color = "#22223b"
        self.text_color = "#f2e9e4"
        self.accent_color = "#4a4e69"
        self.button_hover = "#9a8c98"
        self.root.configure(bg=self.bg_color)

        self.image_data = {}
        self.stages = (
            "1. Raw Thermal Input",
            "2. FFT Spectrum (Tilt Check)",
            "3. LTI Variance Map",
            "4. Cell Rect Mask",
            "5. Panel ROI",
            "6. Thermal Delta Map",
            "7. Verified Faults + Wavelet",
            "8. Final Annotated Output"
        )
        self.current_stage = tk.StringVar(value=self.stages[0])
        self.current_faults = []
        
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
        style.configure("Treeview", background=self.panel_color, foreground=self.text_color, fieldbackground=self.panel_color, borderwidth=0)
        style.configure("Treeview.Heading", background=self.accent_color, foreground="white", font=("Arial", 10, "bold"))
        style.map("Treeview", background=[('selected', self.button_hover)])

    def setup_ui(self):
        # ── Left Sidebar (Controls & Data) ──
        sidebar = tk.Frame(self.root, bg=self.panel_color, width=350, padx=20, pady=20)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Header
        tk.Label(sidebar, text="PV Inspection Panel", font=("Arial", 16, "bold"), bg=self.panel_color, fg="white").pack(pady=(0, 20))

        # Upload Button
        upload_btn = tk.Button(sidebar, text="📁 Upload Thermal Image", command=self.process_image, 
                               bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), relief="flat", pady=10)
        upload_btn.pack(fill=tk.X, pady=(0, 20))

        # ── Upgraded Stage Navigator ──
        tk.Label(sidebar, text="Pipeline Stage:", font=("Arial", 10, "bold"), bg=self.panel_color, fg="#c9ada7").pack(anchor="w", pady=(5, 5))
        
        nav_frame = tk.Frame(sidebar, bg=self.panel_color)
        nav_frame.pack(fill=tk.X, pady=(0, 25))
        
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

        # ── Metadata Section ──
        tk.Label(sidebar, text="Scan Diagnostics:", font=("Arial", 12, "bold"), bg=self.panel_color, fg="white").pack(anchor="w", pady=(10, 5))
        self.meta_text = tk.StringVar(value="Waiting for image...\n\n\n")
        tk.Label(sidebar, textvariable=self.meta_text, justify=tk.LEFT, font=("Consolas", 10), bg=self.panel_color, fg="#9a8c98").pack(anchor="w", pady=(0, 20))

        # ── Fault Table ──
        tk.Label(sidebar, text="Detected Faults (Ranked by FSI):", font=("Arial", 12, "bold"), bg=self.panel_color, fg="white").pack(anchor="w", pady=(0, 5))
        
        columns = ("Type", "Conf", "FSI")
        self.tree = ttk.Treeview(sidebar, columns=columns, show="headings", height=10)
        self.tree.heading("Type", text="Fault Type")
        self.tree.heading("Conf", text="Conf %")
        self.tree.heading("FSI", text="FSI Score")
        self.tree.column("Type", width=120)
        self.tree.column("Conf", width=60, anchor=tk.CENTER)
        self.tree.column("FSI", width=70, anchor=tk.CENTER)
        self.tree.pack(fill=tk.X, pady=(0, 20))

        # Export Button
        self.export_btn = tk.Button(sidebar, text="💾 Export Fault CSV", command=self.export_csv, 
                               bg=self.accent_color, fg="white", font=("Arial", 10, "bold"), relief="flat", state=tk.DISABLED)
        self.export_btn.pack(fill=tk.X)

        # ── Right Main Area (Visualizer) ──
        main_area = tk.Frame(self.root, bg=self.bg_color)
        main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
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

    def process_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Local Test Data",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not file_path:
            return

        try:
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Run Pipeline via final1.py
            img_bgr = cv2.imread(file_path)
            gray_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray_raw.shape
            
            angle, gray_aligned, rot_mat, fft_mag = final1.detect_and_align_tilt(gray_raw)
            std_map, low_var_mask = final1.generate_cell_texture_mask(gray_aligned)
            rect_boxes_mask, border_overlay, panel_roi = final1.build_rect_joined_roi_with_padding(low_var_mask, std_map)
            wav_energy = final1.wavelet_enhance_thermal(gray_aligned)
            thermal_delta, percentage_map, faults, max_p_bright = final1.detect_faults_inside_roi(gray_aligned, panel_roi, wav_energy)

            self.current_faults = faults 

            pct_display = cv2.normalize(percentage_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            wav_display = cv2.normalize(wav_energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            overlay_rgb = np.stack([pct_display, wav_display, np.zeros_like(pct_display)], axis=-1)

            inv_rot_mat = cv2.invertAffineTransform(rot_mat)
            annotated = img_bgr.copy()
            panel_roi_orig = cv2.warpAffine(panel_roi, inv_rot_mat, (w, h), flags=cv2.INTER_NEAREST)
            panel_cnts, _ = cv2.findContours(panel_roi_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, panel_cnts, -1, (0, 255, 0), 2)

            counts = {"Hotspot": 0, "Line Fault": 0, "Partial Shading": 0}

            for fault in faults:
                counts[fault['type']] += 1
                x, y, bw, bh = fault['box']
                pts_aligned = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float32)
                pts_homo = np.hstack([pts_aligned, np.ones((4, 1), dtype=np.float32)])
                pts_orig = np.dot(pts_homo, inv_rot_mat.T).astype(np.int32)
                cv2.polylines(annotated, [pts_orig], isClosed=True, color=fault['color'], thickness=2)
                
                self.tree.insert("", tk.END, values=(fault['type'], fault['confidence'], fault['fsi']))

            total = sum(counts.values())
            meta_info = f"Tilt Corrected: {angle:.1f}°\n"
            meta_info += f"Max Brightness: {max_p_bright:.1f}\n"
            meta_info += f"Total Faults:   {total}\n"
            meta_info += f" - Hotspots:    {counts['Hotspot']}\n"
            meta_info += f" - Line Faults: {counts['Line Fault']}\n"
            meta_info += f" - Partial Shading: {counts['Partial Shading']}"
            self.meta_text.set(meta_info)

            self.export_btn.config(state=tk.NORMAL)

            self.image_data = {
                "1. Raw Thermal Input": (gray_raw, 'gray'),
                "2. FFT Spectrum (Tilt Check)": (fft_mag, 'inferno'),
                "3. LTI Variance Map": (std_map, 'viridis'),
                "4. Cell Rect Mask": (rect_boxes_mask, 'gray'),
                "5. Panel ROI": (cv2.cvtColor(border_overlay, cv2.COLOR_BGR2RGB), None),
                "6. Thermal Delta Map": (thermal_delta, 'magma'),
                "7. Verified Faults + Wavelet": (overlay_rgb, None),
                "8. Final Annotated Output": (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), None)
            }

            self.current_stage.set(self.stages[0])
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")

    def update_display(self, event=None):
        if not self.image_data:
            return

        stage_key = self.current_stage.get()
        current_idx = self.stages.index(stage_key)
        
        # Dynamically enable/disable navigation buttons based on current stage
        self.prev_btn.config(state=tk.NORMAL if current_idx > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if current_idx < len(self.stages) - 1 else tk.DISABLED)

        img, cmap = self.image_data.get(stage_key, (None, None))

        self.ax.clear()
        self.ax.axis('off')
        
        if img is not None:
            if cmap:
                self.ax.imshow(img, cmap=cmap)
            else:
                self.ax.imshow(img)
            
            self.ax.set_title(stage_key, color="white", fontsize=14, fontweight="bold", pad=15)
            
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
            with open(save_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Fault Type", "Confidence (%)", "FSI Score", "IIR Confirmed", "Bounding Box (Rotated x, y, w, h)"])
                for f in self.current_faults:
                    writer.writerow([f['type'], f['confidence'], f['fsi'], f['iir_confirmed'], f['box']])
            messagebox.showinfo("Success", f"Report saved to:\n{os.path.basename(save_path)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalPVDashboard(root)
    root.mainloop()
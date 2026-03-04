import os
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
TRY8_PATH = os.path.join(PROJECT_DIR, "Try8.py")  # same folder as this GUI
INPUT_DIR = os.path.join(PROJECT_DIR, "Input")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "Output")
WORKING_DIR = os.path.join(PROJECT_DIR, "Workingdir")
SINOGRAM_DIR = os.path.join(OUTPUT_DIR, "Sinograms")
TORTURE_DIR = os.path.join(WORKING_DIR, "Torture")


def ensure_dirs():
    for d in [INPUT_DIR, OUTPUT_DIR, WORKING_DIR, SINOGRAM_DIR, TORTURE_DIR]:
        os.makedirs(d, exist_ok=True)


def patch_try8(py_path: str, replacements: dict[str, str]) -> str:
    """
    Replace top-level assignments like:
      filename = "xxx"
      ReconstructionWidth = 64
    with GUI values.
    Returns original text for restore.
    """
    with open(py_path, "r", encoding="utf-8") as f:
        original = f.read()

    patched = original

    # Safe, simple line-based replace using regex for assignments at start of line.
    for var, new_value_literal in replacements.items():
        # matches: var = anything (until end of line)
        pattern = rf"^({re.escape(var)}\s*=\s*).*$"
        repl = rf"\1{new_value_literal}"
        patched, n = re.subn(pattern, repl, patched, flags=re.MULTILINE)
        if n == 0:
            # Not fatal; just warn later in GUI log
            pass

    with open(py_path, "w", encoding="utf-8") as f:
        f.write(patched)

    return original


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SOIL SIRT Runner (Try8) - Minimal GUI")
        self.geometry("900x600")

        self.selected_image_path = tk.StringVar(value="")
        self.width_var = tk.StringVar(value="64")
        self.iter_var = tk.StringVar(value="100")
        self.fan_angle_var = tk.StringVar(value="80")
        self.sourceloc_var = tk.StringVar(value="30")
        self.detectors_var = tk.StringVar(value="55")
        self.plotmatlab_var = tk.BooleanVar(value=False)  # default False for speed

        self._build_ui()

    def _build_ui(self):
        frm = tk.Frame(self)
        frm.pack(fill="x", padx=12, pady=10)

        # File picker
        tk.Label(frm, text="Input image:").grid(row=0, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.selected_image_path, width=70).grid(row=0, column=1, padx=8)
        tk.Button(frm, text="Browse...", command=self.pick_file).grid(row=0, column=2)

        # Parameters
        row = 1
        for label, var in [
            ("ReconstructionWidth", self.width_var),
            ("Iterations", self.iter_var),
            ("Fan angle (deg)", self.fan_angle_var),
            ("Source loc (sourceloc)", self.sourceloc_var),
            ("Detector radius (detectors)", self.detectors_var),
        ]:
            tk.Label(frm, text=label + ":").grid(row=row, column=0, sticky="w", pady=4)
            tk.Entry(frm, textvariable=var, width=20).grid(row=row, column=1, sticky="w", pady=4)
            row += 1

        tk.Checkbutton(frm, text="Enable plotmatlab (slow; makes scan GIF frames)", variable=self.plotmatlab_var)\
            .grid(row=row, column=1, sticky="w", pady=4)
        row += 1

        tk.Button(frm, text="Run Try8", command=self.run).grid(row=row, column=1, sticky="w", pady=8)

        # Log box
        self.log = tk.Text(self, height=22)
        self.log.pack(fill="both", expand=True, padx=12, pady=10)
        self._log("Ready.\n")

    def _log(self, msg: str):
        self.log.insert("end", msg)
        self.log.see("end")
        self.update_idletasks()

    def pick_file(self):
        path = filedialog.askopenfilename(
            title="Choose an input image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if path:
            self.selected_image_path.set(path)

    def run(self):
        ensure_dirs()

        img_path = self.selected_image_path.get().strip()
        if not img_path or not os.path.exists(img_path):
            messagebox.showerror("Missing file", "Please choose a valid input image.")
            return

        # Copy to Input/ with original basename
        basename = os.path.basename(img_path)
        dest_in = os.path.join(INPUT_DIR, basename)
        try:
            # Only copy if source and destination are different
            src = os.path.normpath(os.path.abspath(img_path))
            dst = os.path.normpath(os.path.abspath(dest_in))

            same = False
            try:
                same = os.path.samefile(src, dst)
            except FileNotFoundError:
                same = False
            except OSError:
    # On some setups samefile may fail; fall back to normalized compare
                same = (src.lower() == dst.lower())

            if not same:
                shutil.copy2(src, dst)
            else:
                self._log("Image already in Input folder, skipping copy.\n")
        except Exception as e:
            messagebox.showerror("Copy failed", f"Could not copy image to Input/: {e}")
            return

        # Collect params
        try:
            width = int(self.width_var.get())
            iters = int(self.iter_var.get())
            fan_angle = float(self.fan_angle_var.get())
            sourceloc = float(self.sourceloc_var.get())
            det_radius = float(self.detectors_var.get())
        except ValueError:
            messagebox.showerror("Bad input", "Width/Iterations must be integers; angles/locations must be numbers.")
            return

        plotmatlab = self.plotmatlab_var.get()

        self._log("\n=== Running Try8 ===\n")
        self._log(f"Input: {basename}\n")
        self._log(f"Width={width}, Iterations={iters}, FanAngle={fan_angle}, Sourceloc={sourceloc}, DetRadius={det_radius}\n")
        self._log(f"plotmatlab={plotmatlab}\n")

        # Run in background thread
        threading.Thread(
            target=self._run_worker,
            args=(basename, width, iters, fan_angle, sourceloc, det_radius, plotmatlab),
            daemon=True
        ).start()

    def _run_worker(self, basename, width, iters, fan_angle, sourceloc, det_radius, plotmatlab):
        # Patch Try8.py (temporary)
        try:
            original = patch_try8(
                TRY8_PATH,
                {
                    "filename": repr(basename),                # "xxx.png"
                    "ReconstructionWidth": str(width),         # 64
                    "plotmatlab": "True" if plotmatlab else "False",
                }
            )

            # Also patch the specific call in main():
            # ScanningResult.CreateCurvedFanRays(sourceloc=30, detectors=55, fan_angle_degrees=80)
            # We replace numbers inside that call.
            with open(TRY8_PATH, "r", encoding="utf-8") as f:
                text = f.read()

            text2 = re.sub(
                r"CreateCurvedFanRays\s*\(\s*sourceloc\s*=\s*[^,]+,\s*detectors\s*=\s*[^,]+,\s*fan_angle_degrees\s*=\s*[^)]+\)",
                f"CreateCurvedFanRays(sourceloc={sourceloc}, detectors={det_radius}, fan_angle_degrees={fan_angle})",
                text
            )
            # Patch iteration count at:
            # Reconstuctor.SetIterations(100)
            text2 = re.sub(
                r"Reconstuctor\.SetIterations\s*\(\s*\d+\s*\)",
                f"Reconstuctor.SetIterations({iters})",
                text2
            )

            with open(TRY8_PATH, "w", encoding="utf-8") as f:
                f.write(text2)

            # Run Try8.py and stream logs
            cmd = [sys.executable, TRY8_PATH]
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in proc.stdout:
                self._log(line)

            rc = proc.wait()
            self._log(f"\nProcess finished with code {rc}\n")

            if rc == 0:
                self._log(f"Check outputs in: {OUTPUT_DIR}\n")
                self._log(f"Sinograms in: {SINOGRAM_DIR}\n")
                messagebox.showinfo("Done", "Try8 finished. Check Output/ and Output/Sinograms/")
            else:
                messagebox.showerror("Error", "Try8 failed. See log output.")

        except Exception as e:
            self._log(f"\n[GUI ERROR] {e}\n")
            messagebox.showerror("GUI Error", str(e))

        finally:
            # Restore original Try8.py
            try:
                if "original" in locals():
                    with open(TRY8_PATH, "w", encoding="utf-8") as f:
                        f.write(original)
                    self._log("\n(Restored Try8.py)\n")
            except Exception as e:
                self._log(f"\n[WARN] Failed to restore Try8.py: {e}\n")


if __name__ == "__main__":
    ensure_dirs()
    app = App()
    app.mainloop()
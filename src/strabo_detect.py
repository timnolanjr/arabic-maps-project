#!/usr/bin/env python3
# src/strabo_detect.py
# ------------------------------------------------------------
# Wrapper script to run Strabo text‑recognition on one image or a folder.
# Supports two OCR methods: Strabo's built‑in pipeline or pytesseract fallback.
# Automatically locates run_command_line.py under the repo_root.
# ------------------------------------------------------------

import os
import sys
import argparse
import subprocess
import shutil

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None


def collect_images(path):
    if os.path.isdir(path):
        exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        return [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.splitext(f.lower())[1] in exts
        ]
    elif os.path.isfile(path):
        return [path]
    else:
        raise FileNotFoundError(f"Not a file or folder: {path}")


def find_run_script(repo_root):
    """Search for run_command_line.py under repo_root."""
    for root, _, files in os.walk(repo_root):
        if 'run_command_line.py' in files:
            return os.path.join(root, 'run_command_line.py')
    raise FileNotFoundError(f"Could not find run_command_line.py under {repo_root}")


def run_strabo_on_image(image_path, repo_root, checkpoint_path, config_path, temp_results_dir, ocr_method):
    """
    Calls Strabo's run_command_line.py for detection, then optionally runs pytesseract.
    Outputs go to repo_root/static/results/*; we then move them into temp_results_dir.
    """
    # Locate the Strabo driver script
    run_py = find_run_script(repo_root)
    print(f"Using driver script: {run_py}")

    # Clear previous results
    results_base = os.path.join(repo_root, "static", "results")
    if os.path.isdir(results_base):
        shutil.rmtree(results_base)
    os.makedirs(results_base, exist_ok=True)

    # Build command
    cmd = [
        sys.executable,
        run_py,
        "--checkpoint-path", checkpoint_path,
        "--image", image_path,
        "--config", config_path
    ]
    print("Running Strabo:", " ".join(cmd))

    # Prepare environment
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    # GEOS for Shapely
    for prefix in ["/usr/local/opt/geos/lib", "/opt/homebrew/opt/geos/lib"]:
        lib = os.path.join(prefix, "libgeos_c.dylib")
        if os.path.exists(lib):
            env["GEOS_LIBRARY_PATH"] = lib
            env["DYLD_LIBRARY_PATH"] = ":".join([prefix, env.get("DYLD_LIBRARY_PATH", "")])
            break

    subprocess.run(cmd, check=True, env=env)

    # Collect results
    subdirs = os.listdir(results_base)
    if len(subdirs) != 1:
        raise RuntimeError(f"Expected 1 results subdir, found: {subdirs}")
    result_subdir = os.path.join(results_base, subdirs[0])

    base = os.path.splitext(os.path.basename(image_path))[0]
    dest = os.path.join(temp_results_dir, base)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.move(result_subdir, dest)
    print(f"→ Strabo output for {image_path} moved to {dest}")

    # Fallback OCR
    if ocr_method == 'pytesseract':
        if pytesseract is None:
            raise RuntimeError("pytesseract not installed; cannot use fallback OCR.")
        final_img = os.path.join(dest, 'output.png')
        if not os.path.isfile(final_img):
            raise FileNotFoundError(f"No output.png found in {dest}")
        print("Running pytesseract fallback on", final_img)
        img = Image.open(final_img)
        text = pytesseract.image_to_string(img, lang='ara')
        txt_path = os.path.join(dest, f"{base}_pytesseract.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"→ Fallback OCR saved to {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Strabo on image(s) and collect results"
    )
    parser.add_argument("--repo_root", required=True,
                        help="Path to strabo-text-recognition-deep-learning repo")
    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to east_icdar2015_resnet_v1_50_rbox folder")
    parser.add_argument("--config", required=True,
                        help="Path to Strabo configuration.ini")
    parser.add_argument("--input", required=True,
                        help="Image file or folder to process")
    parser.add_argument("--output_dir", default="./strabo_results/",
                        help="Where to collect Strabo result folders")
    parser.add_argument("--ocr_method", choices=['strabo','pytesseract'],
                        default='strabo',
                        help="OCR method to use")
    args = parser.parse_args()

    # Absolutize
    args.repo_root = os.path.abspath(args.repo_root)
    args.checkpoint_path = os.path.abspath(args.checkpoint_path)
    args.config = os.path.abspath(args.config)
    args.input = os.path.abspath(args.input)
    args.output_dir = os.path.abspath(args.output_dir)

    # Validate
    for p in (args.repo_root, args.checkpoint_path, args.config):
        if not os.path.exists(p):
            parser.error(f"Path does not exist: {p}")
    os.makedirs(args.output_dir, exist_ok=True)

    images = collect_images(args.input)
    for img in images:
        run_strabo_on_image(
            img, args.repo_root, args.checkpoint_path,
            args.config, args.output_dir, args.ocr_method
        )

if __name__ == "__main__":
    main()

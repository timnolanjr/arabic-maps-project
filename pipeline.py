from pathlib import Path
import argparse
from src.pipeline_core import run_pipeline

def main():
    ap = argparse.ArgumentParser(description="Unified geometry pipeline")
    ap.add_argument("input", help="Image file or directory")
    ap.add_argument("-o", "--out", default="data/processed_maps", help="Output base directory")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    run_pipeline(Path(args.input), Path(args.out), interactive=args.interactive)

if __name__ == "__main__":
    main()

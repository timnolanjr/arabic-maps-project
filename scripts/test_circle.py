#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.circle import interactive_detect_and_save, batch_detect_and_save

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Path to input image")
    p.add_argument("-o","--output-dir", default="data/processed_maps")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--num-candidates", type=int, default=100)
    args = p.parse_args()

    img = Path(args.input)
    out = Path(args.output_dir)
    if args.interactive:
        interactive_detect_and_save(img, out)
    else:
        batch_detect_and_save(img, out, args.num_candidates)

if __name__=="__main__":
    main()

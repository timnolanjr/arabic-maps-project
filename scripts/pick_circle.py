#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.circle import review_candidates_and_save

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Path to input image")
    p.add_argument("-o","--output-dir", default="data/processed_maps")
    args = p.parse_args()

    review_candidates_and_save(Path(args.input), Path(args.output_dir))

if __name__=="__main__":
    main()

import argparse
import os
import cv2 as cv
from tqdm import tqdm

from . import LAYOUTS_OK, VALID_EXTS, key_from
from .circle import detect_circle, human_size
from .sheet import fetch_sheet, save_metadata, upload_to_drive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', required=True)
    parser.add_argument('-o', '--output-dir', default='Maps_circleGuess')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    layout_map, done = fetch_sheet()
    files = [f for f in os.listdir(args.input_dir)
             if f.lower().endswith(VALID_EXTS)]
    files.sort(key=lambda f: os.path.getsize(os.path.join(args.input_dir, f)))
    print(f"[   ] {len(files)} files queued (filtered + sorted by size)\n")

    os.makedirs(args.output_dir, exist_ok=True)
    for fn in tqdm(files, desc="Files"):
        key = key_from(fn)
        path = os.path.join(args.input_dir, fn)
        sz = os.path.getsize(path)

        if layout_map.get(key, '') not in LAYOUTS_OK:
            print(f"\n skipping {fn} ({human_size(sz)}): layout not allowed")
            continue
        if key in done:
            ans = input(f"\n{fn} ({human_size(sz)}) already done—skip? (y/n): ")
            if ans.lower().startswith('y'):
                print(" skipped\n")
                continue

        print(f"\n→ Processing {fn} ({human_size(sz)})")
        res = detect_circle(path, args.interactive)
        if not res or res == 'skip':
            print(" skipped\n")
            continue

        x, y, r = res
        save_metadata(key, x, y, r, layout_map[key])

        src = cv.imread(path)
        cv.circle(src, (int(x), int(y)), int(r), (255, 0, 255), 3)
        cv.circle(src, (int(x), int(y)), max(3, int(r*0.01)), (255, 0, 255), 3)
        out = os.path.join(args.output_dir, fn)
        cv.imwrite(out, src)

        upload_to_drive(out, fn)

        if args.show:
            import matplotlib.pyplot as plt
            plt.imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))
            plt.title(fn)
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    main()

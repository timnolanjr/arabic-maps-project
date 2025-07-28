# scripts/test_circle.py

import sys
import os
# Ensure src directory is on the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from circle import (
    median_blur_and_gray,
    detect_circle_hough,
    draw_multiple_circles,
    draw_circle,
    save_circle_params,
    get_output_paths
)


def main():
    # Define input and base output directory
    input_path = Path("/Users/tim/Documents/Projects/arabic-maps-project/data/Raw_Maps/al-Qazwīnī_Arabic MSS 575.jpg")
    base_output_dir = Path("data/Processed_Maps")

    # Load and preprocess image
    img = cv2.imread(str(input_path))
    gray = median_blur_and_gray(img)

    # Detect candidate circles
    candidates = detect_circle_hough(gray, top_k=5)
    if not candidates:
        print("No circles detected.")
        return

    # Get paths
    paths = get_output_paths(input_path, base_output_dir)

    # Save and display all candidates
    guess_img = draw_multiple_circles(img, candidates)
    cv2.imwrite(str(paths["circle_guesses_img"]), guess_img)
    plt.imshow(cv2.cvtColor(guess_img, cv2.COLOR_BGR2RGB))
    plt.title("Top Circle Candidates")
    plt.axis('off')
    plt.show()

    # Prompt user to pick best circle
    for i, (cx, cy, r) in enumerate(candidates, 1):
        print(f"[{i}] center=({cx}, {cy}), radius={r}")
    choice = int(input(f"Select best circle [1-{len(candidates)}]: "))
    cx, cy, r = candidates[choice - 1]

    # Save final result
    final_img = draw_circle(img, cx, cy, r)
    cv2.imwrite(str(paths["circle_final_img"]), final_img)
    save_circle_params(paths["circle_params_json"], cx, cy, r)

    print(f"Final circle saved to {paths['circle_final_img']}")
    print(f"Parameters saved to {paths['circle_params_json']}")


if __name__ == "__main__":
    main()

import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np

def click_points(img: np.ndarray, n: int = 8) -> List[Tuple[float,float]]:
    """
    Display image, collect n clicks, then block until window is closed.
    Returns list of (x,y).
    """
    disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(disp)
    ax.set_title(f"Click {n} points. Close the figure to continue execution.")
    ax.axis("off")

    plt.show(block=False)
    pts = plt.ginput(n, timeout=-1)

    # wait until user closes the window
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)
    plt.close(fig)
    return pts

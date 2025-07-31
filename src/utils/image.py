import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import Tuple, List

def median_blur_and_gray(bgr_img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Convert BGR image to grayscale and apply median blur.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return cv2.medianBlur(gray, ksize)

def show_image(img: np.ndarray, title: str = None):
    """
    Display a BGR OpenCV image in Matplotlib  with optional blocking window.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    if title:
        ax.set_title(title)
    ax.axis("off")
    plt.show(block=True)
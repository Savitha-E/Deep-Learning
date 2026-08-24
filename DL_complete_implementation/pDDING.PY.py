import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

image_path = "/home/ibab/Pictures/IMG-20241002-WA0056.jpg"


def padding(image_path, padding=2, display=True):
    image_array = cv2.imread(image_path)
    if len(image_array.shape) == 3:
        H, W, C = image_array.shape
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    elif len(image_array.shape) == 2:
        H, W = image_array.shape

    new_h = H + 2 * padding
    new_w = W + 2 * padding

    if C:
        padded_image = np.zeros((new_h, new_w, C), dtype=image_array.dtype)
        padded_image[padding:padding + H, padding:padding + W, :] = image_array
    else:
        padded_image = np.zeros((new_h, new_w), dtype=image_array.dtype)
        padded_image[padding:padding + H, padding:padding + W] = image_array

    if display:
        plt.subplot(1, 2, 1)
        plt.imshow(image_array, cmap="gray" if C is None else None)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(padded_image, cmap="gray" if C is None else None)
        plt.axis("off")

        plt.show()

    return padded_image


padding(image_path, padding=20)

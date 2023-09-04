import cv2
import numpy as np
import matplotlib.pyplot as plt
def make_grey_image(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return  gray_image


img = plt.imread('skull.jpg')
gray_image = make_grey_image(img)
[height, width] = gray_image.shape
sampled_image = []
sampled_image.append(gray_image.copy())

for k in range(7):
    for i in range(height):
        for j in range(width):
            gray_image[i][j] = gray_image[i][j] >> 1

    sampled_image.append(gray_image.copy())
row, col = 2, 4
idx = 0
fig, ax = plt.subplots(row, col, figsize=(9, 7))

for i in range(row):
    for j in range(col):
        ax[i, j].imshow(sampled_image[idx], cmap='gray')
        ax[i, j].set_title(f'{8-idx}bits')
        idx+=1

plt.tight_layout()
plt.show()



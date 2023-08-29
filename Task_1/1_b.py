import cv2
import numpy as np
img = cv2.imread('monalisa.png', 0)
cv2.imshow('Original Image', img)
cv2.waitKey(0)

[height, width] = img.shape
print(height)
print(width)

for i in range(height):
    for j in range(width):
        img[i][j] = img[i][j] >> 1



cv2.imshow('Image', img)
cv2.waitKey(0)

cv2.destroyAllWindows()

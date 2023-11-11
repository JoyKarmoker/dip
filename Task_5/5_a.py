import matplotlib.pyplot as plt
import numpy as np
import cv2

def perform_erosion(image, structuring_element_size):
    row, cols = image.shape
    eroded_image = np.zeros(image.shape)

    for i in range(structuring_element_size//2, row-structuring_element_size//2):
        for j in range(structuring_element_size//2, cols-structuring_element_size//2):
            region = image[i-structuring_element_size//2:i+structuring_element_size//2+1,
                     j-structuring_element_size//2:j+structuring_element_size//2+1]
            eroded_image[i, j] = np.min(region)

    return eroded_image

def perform_dilation(image, structuring_element_size):
    row, cols = image.shape
    eroded_image = np.zeros(image.shape)

    for i in range(structuring_element_size//2, row-structuring_element_size//2):
        for j in range(structuring_element_size//2, cols-structuring_element_size//2):
            region = image[i-structuring_element_size//2:i+structuring_element_size//2+1,
                     j-structuring_element_size//2:j+structuring_element_size//2+1]
            eroded_image[i, j] = np.max(region)

    return eroded_image


original_image = cv2.imread('wirebond.tif', cv2.IMREAD_GRAYSCALE)
structuring_element_size = 15
eroded_image = perform_erosion(original_image, structuring_element_size)
dilated_image = perform_dilation(original_image, structuring_element_size)

plt.figure(figsize=(8, 8))
plt.subplot(2, 2, (1, 2))
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.subplot(2, 2, 3)
plt.imshow(eroded_image, cmap='gray')
plt.title('Eroded Image')
plt.subplot(2, 2, 4)
plt.imshow(dilated_image, cmap='gray')
plt.title('Dilated Image')

plt.tight_layout()
plt.show()

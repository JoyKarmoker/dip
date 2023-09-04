import numpy as np
import matplotlib.pyplot as plt

def convert_to_gray(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image


def find_histogram_values(gray_image):
    pixels_values = np.zeros(256, dtype=int)
    [height, width] = gray_image.shape
    for i in range(height):
        for j in range(width):
            pixels_values[gray_image[i][j]] +=1
    return pixels_values

def segment_image(gray_image, threshold):
    image = gray_image.copy()
    [height, width] = gray_image.shape
    for i in range(height):
        for j in range(width):
            if(image[i][j] < threshold):
                image[i][j] = 0
            else:
                image[i][j] = 255
    return image


image = plt.imread('lena.jpg')
gray_image = convert_to_gray(image)
pixel_values = find_histogram_values(gray_image)
threshold = 100
segmented_image = segment_image(gray_image, threshold)

plt.figure(figsize=(9,7))
plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(gray_image, cmap='gray')
plt.subplot(2,2,2)
plt.bar(range(256), pixel_values)
plt.title("Histogram")
plt.subplot(2,2,(3,4))
plt.imshow(segmented_image, cmap="gray")
plt.title(f'binary Threshold:{threshold}')
plt.show()

import numpy as np
import matplotlib.pyplot as plt


def make_gray_image(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image


def histogram_generate(image):
    pixel_counts = np.zeros(256, dtype=int)
    for row in image:
        for pixel_value in row:
            pixel_counts[pixel_value] += 1

    return pixel_counts

rgb_image = plt.imread('images/lena.jpg')
gray_image = make_gray_image(rgb_image)

pixel_counts = histogram_generate(gray_image)
# print(pixel_counts)


threshold = 100
segmented_image = (gray_image > threshold).astype(np.uint8)*255
# print(segmented_image)

plt.figure(figsize=(8, 7))
plt.subplot(2,2,1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')
plt.subplot(2,2,2)
plt.bar(range(256), pixel_counts)
plt.title('Histogram')
plt.subplot(2,2,(3,4))
plt.imshow(segmented_image, cmap='gray')
plt.title(f'binary Threshold:{threshold}')

plt.tight_layout()
plt.show()

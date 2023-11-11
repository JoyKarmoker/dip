import matplotlib.pyplot as plt
import numpy as np


def add_gaussian_noise(image, mean=0, stddev=0.5):
    gaussian_noise = np.random.normal(mean, stddev, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


original_image = plt.imread('pattern.tif')
noisy_image = add_gaussian_noise(original_image, 0, 50)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.subplot(2, 1, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.show()

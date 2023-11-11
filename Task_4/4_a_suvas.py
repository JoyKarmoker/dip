import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def add_gaussian_noise(image, mean=0, std=1):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image+noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def batterworth_low_pass_filter(image, order, cut_off_frequency):
    height, width = image.shape
    my_filter = np.zeros(image.shape, dtype=np.float32)

    frequncy_domain_image = np.fft.fft2(image)
    frequncy_domain_image = np.fft.fftshift(frequncy_domain_image)
    n = order
    d0 = cut_off_frequency

    for i in range(height):
        for j in range(width):
            d = np.sqrt((i - height / 2) ** 2 + (j - width / 2) ** 2)
            my_filter[i, j] = 1 / (1 + (d / d0) ** (2 * n))

    filteredImage = frequncy_domain_image * my_filter
    filteredImage = np.fft.fftshift(filteredImage)
    filteredImage = np.abs(np.fft.ifft2(filteredImage))

    return filteredImage


def gaussion_low_pass_filter(image, cutoff):
    height, width = image.shape
    my_filter = np.zeros(image.shape, dtype=np.float32)

    frequency_domain_image = np.fft.fft2(image)
    frequency_domain_image = np.fft.fftshift(frequency_domain_image)

    d0 = cutoff

    for i in range(height):
        for j in range(width):
            d = np.sqrt((i - height / 2) ** 2 + (j - width / 2) ** 2)
            my_filter[i, j] = np.exp(-(d ** 2) / (2 * d0 ** 2))

    filteredImage = frequency_domain_image * my_filter

    filteredImage = np.fft.fftshift(filteredImage)
    filteredImage = np.abs(np.fft.ifft2(filteredImage))

    return filteredImage




# opening image
original_image = plt.imread('pattern.tif').copy()
gaussian_noisy_image = add_gaussian_noise(original_image, mean=0, std=125)
butterFilteredImage = batterworth_low_pass_filter(original_image, 2, 160)
gaussianFilteredImage = gaussion_low_pass_filter(original_image, 10)


plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.subplot(2, 2, 2)
plt.imshow(gaussian_noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.tight_layout()
plt.subplot(2, 2, 3)
plt.imshow(butterFilteredImage, cmap='gray')
plt.title('Butterworth Filter')
plt.subplot(2, 2, 4)
plt.imshow(gaussianFilteredImage, cmap="gray")
plt.title('Gaussian Filter')
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt


def add_gaussian_noise(img, mean=0, stddev=1):
    gaussian_noise = np.random.normal(mean, stddev, img.shape)
    noisy_image = img + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


def butterworth_low_pass_filter(image, order, cut_off_frequency):
    height, width = image.shape
    H = np.zeros(image.shape, dtype=np.float32)

    # frequncy_domain_image = np.fft.fft2(image)
    # frequncy_domain_image = np.fft.fftshift(frequncy_domain_image)
    frequency_domain_image = np.fft.fftshift(np.fft.fft2(image))
    n = order
    d0 = cut_off_frequency

    for i in range(height):
        for j in range(width):
            d = np.sqrt((i - height / 2) ** 2 + (j - width / 2) ** 2)
            H[i, j] = 1 / (1 + (d / d0) ** (2 * n))

    filteredImage = frequency_domain_image * H
    filteredImage = np.abs(np.fft.ifft2(filteredImage))
    filteredImage = filteredImage / 255
    return filteredImage


def gaussian_low_pass_filter(image, cut_off_frequency):
    frequency_domain_image = np.fft.fftshift(np.fft.fft2(image))

    D0 = cut_off_frequency
    height, width = image.shape
    H = np.zeros(image.shape, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            d = np.sqrt((i-height/2)**2 + (j-width/2)**2)
            H[i, j] = np.exp(-(d**2) / (2*D0)**2)

    filtered_image = frequency_domain_image * H
    filtered_image = np.abs(np.fft.ifft2(filtered_image))
    filtered_image = filtered_image / 255
    return filtered_image


img = plt.imread('pattern.tif')
noisy_image = add_gaussian_noise(img, mean=0, stddev= 50)
b_filtered_image = butterworth_low_pass_filter(noisy_image, 4, 40)
g_filtered_image = gaussian_low_pass_filter(noisy_image, 40)


plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(2, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.tight_layout()
plt.subplot(2, 2, 3)
plt.imshow(b_filtered_image, cmap='gray')
plt.title('Butterworth Filter')
plt.subplot(2, 2, 4)
plt.imshow(g_filtered_image, cmap='gray')
plt.title('Gaussian Filter')
plt.show()

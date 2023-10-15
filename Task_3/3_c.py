import numpy as np
import matplotlib.pyplot as plt
import random


def convert_to_gray(image):
    if len(image.shape) == 2:
        return image
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image


def add_salt_peper_noise(image, noise_percentage):
    noise = noise_percentage / 100
    noisy_image = image.copy()
    row = noisy_image.shape[0]
    column = noisy_image.shape[1]
    total_pixels = row * column
    noise_pixels = int(noise * total_pixels)
    for i in range(noise_pixels):
        random_row_in_original_image = random.randint(0, row - 1)
        random_col_in_original_image = random.randint(0, column - 1)
        noisy_image[random_row_in_original_image][random_col_in_original_image] = random.randint(0, 255)
    return noisy_image


def harmonic_mean_filter(noisy_image, kernel_size):
    filtered_image = np.zeros_like(noisy_image)
    pad_size = kernel_size // 2
    starting_row = 0 + pad_size
    starting_col = 0 + pad_size
    ending_row = filtered_image.shape[0] - pad_size
    ending_col = filtered_image.shape[1] - pad_size

    for i in range(starting_row, ending_row):
        for j in range(starting_col, ending_col):
            window = noisy_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            #Handle zero values by setting them to a small positive value.
            window[window == 0] = 0.01
            num_zeros = np.count_nonzero(window == 0)
            if num_zeros == 0:
                # Calculate the reciprocal of the window.
                reciprocal_window = 1.0 / window
                # Calculate the harmonic mean of the window.
                harmonic_mean = (kernel_size * kernel_size) / np.sum(reciprocal_window)
                # Set the filtered pixel value to the harmonic mean.
                filtered_image[i][j] = harmonic_mean
            else:
                # If there is a divide-by-zero error, set the filtered pixel value to 0.
                filtered_image[i][j] = 0
    return filtered_image

def geometric_mean_filter(noisy_image, kernel_size):
    height, width = noisy_image.shape
    filtered_image = np.zeros_like(noisy_image, dtype=np.float64)
    pad_size = kernel_size // 2

    for i in range(height):
        for j in range(width):
            pixel, count = 1, 0
            for m in range(-pad_size, pad_size + 1, 1):
                for n in range(-pad_size, pad_size + 1, 1):
                    if (i + m >= 0 and i + m < height and j + n >= 0 and j + n < width):
                        if (noisy_image[i + m, j + n] != 0):
                            count += 1
                            pixel = pixel * int(noisy_image[i + m, j + n])
            count = 1 if count == 0 else count
            filtered_image[i][j] = pixel ** (1 / count)
    return filtered_image

def cal_psnr(original_image, noisy_image):
    original_image = original_image.astype(np.float64)
    noisy_image = noisy_image.astype(np.float64)
    mse = np.mean((original_image - noisy_image) ** 2)
    max_pixel_value = 255
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    return psnr


rgb_image = plt.imread('skull.jpg')
gray_image = convert_to_gray(rgb_image)
noise_percentage = 2
noisy_image = add_salt_peper_noise(gray_image, noise_percentage)
noise_psnr = cal_psnr(gray_image, noisy_image)
kernel_size = 3
harmonic_mean_filtered_image = harmonic_mean_filter(noisy_image, kernel_size)
harmonic_psnr = cal_psnr(gray_image, harmonic_mean_filtered_image)
geometric_mean_filtered_image = geometric_mean_filter(noisy_image, kernel_size)
geometric_psnr = cal_psnr(gray_image, geometric_mean_filtered_image)

plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.subplot(2, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title(f'Noisy Image\nPSNR {noise_psnr:.2f} dB')
plt.subplot(2, 2, 3)
plt.imshow(harmonic_mean_filtered_image, cmap='gray')
plt.title(f'Harmonic Mean Filter using {kernel_size} by {kernel_size} kernel\nPSNR {harmonic_psnr:.2f} dB')
plt.subplot(2, 2, 4)
plt.imshow(geometric_mean_filtered_image, cmap='gray')
plt.title(f'Geometric Mean Filter using {kernel_size} by {kernel_size} kernel\nPSNR {geometric_psnr:.2f} dB')

plt.tight_layout()
plt.show()

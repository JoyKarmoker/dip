import numpy as np
import matplotlib.pyplot as plt
import random

def convert_to_gray(image):
    if len(image.shape) == 2:
        return image
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image


def add_salt_peper_noise(image, noise_percentage):
    noise = noise_percentage/100
    noisy_image = image.copy()
    row = noisy_image.shape[0]
    column = noisy_image.shape[1]
    total_pixels = row * column
    noise_pixels = int(noise*total_pixels)
    for i in range(noise_pixels):
        random_row_in_original_image = random.randint(0, row-1)
        random_col_in_original_image = random.randint(0, column-1)
        noisy_image[random_row_in_original_image][random_col_in_original_image] = random.choice([0, 255])
    return noisy_image


def average_filter(noisy_image, kernel_size):
    filtered_image = np.zeros_like(noisy_image)
    pad_size = kernel_size // 2
    starting_row = 0+pad_size
    starting_col = 0+pad_size
    ending_row = filtered_image.shape[0] - pad_size
    ending_col = filtered_image.shape[1] - pad_size
    mask = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

    for i in range(starting_row, ending_row):
        for j in range(starting_col, ending_col):
            window = noisy_image[i - pad_size:i + pad_size + 1, j-pad_size:j+pad_size+1]
            temp_window = window.copy()
            temp_window = temp_window * mask
            temp_window_mean = np.sum(temp_window)
            filtered_image[i][j] = temp_window_mean

    return filtered_image


def median_filter(noisy_image, kernel_size):
    filtered_image = np.zeros_like(noisy_image)
    pad_size = kernel_size // 2
    starting_row = 0+pad_size
    starting_col = 0+pad_size
    ending_row = filtered_image.shape[0] - pad_size
    ending_col = filtered_image.shape[1] - pad_size

    for i in range(starting_row, ending_row):
        for j in range(starting_col, ending_col):
            window = noisy_image[i - pad_size:i + pad_size + 1, j-pad_size:j+pad_size+1]
            window_mean = np.median(window)
            filtered_image[i][j] = window_mean

    return filtered_image 



def cal_psnr(original_image, noisy_image):
    original_image = original_image.astype(np.float64)
    noisy_image = noisy_image.astype(np.float64)
    mse = np.mean((original_image-noisy_image) ** 2)
    max_pixel_value = 255
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    return psnr


rgb_image = plt.imread('moon.jpg')
gray_image = convert_to_gray(rgb_image)
noise_percentage = 10
noisy_image = add_salt_peper_noise(gray_image, noise_percentage)
noise_psnr = cal_psnr(gray_image, noisy_image)
kernel_size = 3
average_filtered_image = average_filter(noisy_image, kernel_size)
average_psnr = cal_psnr(gray_image, average_filtered_image)
median_filtered_image = median_filter(noisy_image, kernel_size)
median_psnr = cal_psnr(gray_image, median_filtered_image)


plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.subplot(2, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title(f'Noisy Image\nPSNR {noise_psnr:.2f} dB')
plt.subplot(2, 2, 3)
plt.imshow(average_filtered_image, cmap='gray')
plt.title(f'Average Filter using {kernel_size} by {kernel_size} kernel\nPSNR {average_psnr:.2f} dB')
plt.subplot(2, 2, 4)
plt.imshow(median_filtered_image, cmap='gray')
plt.title(f'Median Filter using {kernel_size} by {kernel_size} kernel\nPSNR {median_psnr:.2f} dB')

plt.tight_layout() 
plt.show()

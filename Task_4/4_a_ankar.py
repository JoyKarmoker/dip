import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('pattern.tif', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(img, (512, 512))

height, width = image.shape

mean = 0
stddev = 25
noise = np.random.normal(mean, stddev, image.shape)

noisy_image = image + noise

# Compute the 2D discrete Fourier transform (DFT) of the noisy image
F = np.fft.fftshift(np.fft.fft2(noisy_image))

# Define the Butterworth filter

D0 = 10  # Cutoff frequency
n = 10  # Filter order


def gaussian(F):
    M, N = F.shape
    Gaussian = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)

            Gaussian[u, v] = np.exp(-((D ** 2) / (2 * D0 ** 2)))

    Gaussian_constant = Gaussian * F
    filter_image = np.abs(np.fft.ifft2(Gaussian_constant))
    # filtered_image = cv2.normalize(filter_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    filtered_image = filter_image / 255
    return filtered_image


def butterworth(F):
    M, N = F.shape
    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            H[u, v] = 1 / (1 + (D / D0) ** (2 * n))
    G = F * H
    filtered_image = np.abs(np.fft.ifft2(G))
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    return filtered_image


butter_filter = butterworth(F)
gaussian_filter = gaussian(F)
# Display the original noisy image and the filtered image
plt.subplot(2, 2, 1)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')

plt.subplot(2, 2, 3)
plt.imshow(butter_filter, cmap='gray')
plt.title('butter Filtered Image')

plt.subplot(2, 2, 4)
plt.imshow(gaussian_filter, cmap='gray')
plt.title('gaussian Filtered Image')
plt.tight_layout()

plt.show()
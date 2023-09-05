import numpy as np
import matplotlib.pyplot as plt


def convert_to_gray(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image


def power_law(image, gamma):
    f_range = np.arange(0, 256)
    f_value = np.power(f_range, gamma)
    transformed_image = image.copy()
    [height, width] = transformed_image.shape
    transformed_image = transformed_image/255.0
    for i in range(height):
        for j in range(width):
            transformed_image[i, j]  = (transformed_image[i, j] ** gamma) * 255

    return transformed_image, f_value, f_range


def inverse_log(image):
    inverse_log_image = image.copy()
    inverse_log_image = inverse_log_image/255.0
    r = np.arange(0, 256)
    c = 255.0 / np.log(1 + 255)

    # for inverse log operation
    y = (np.exp(r) ** (1/c)) - 1
    [height, width] = inverse_log_image.shape
    for i in range(height):
        for j in range(width):
            inverse_log_image[i][j] = np.exp(inverse_log_image[i][j]*255) ** (1/c) - 1
    

    return inverse_log_image, y, r


rgb_image = plt.imread('landscape.jpg')
gray_image = convert_to_gray(rgb_image)

power = 2
power_law_image, f_value, f_range = power_law(gray_image, power)
log_image, log_f_value, log_f_range = inverse_log(gray_image)


plt.figure(figsize=(8, 7))
plt.subplot(3, 2, (1, 2))
plt.imshow(gray_image, cmap='gray')
plt.title('Original')
plt.subplot(3, 2, 3)
plt.plot(f_range, f_value)
plt.title(f'Transfer Function(power-{power})')
plt.subplot(3, 2, 4)
plt.imshow(power_law_image, cmap='gray')
plt.title('Transformed')
plt.subplot(3, 2, 5)
plt.plot(log_f_range, log_f_value)
plt.title('Transfer Function(inverse-log)')
plt.subplot(3, 2, 6)
plt.imshow(log_image, cmap='gray')
plt.title('Transformed')

plt.tight_layout()
plt.show()

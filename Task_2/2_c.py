import numpy as np
import matplotlib.pyplot as plt
def make_gray_image(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image

def last_three_bits(gray_image):
    converted_image = gray_image.copy()
    [height, width] = converted_image.shape
    for i in range(height):
        for j in range(width):
            converted_image[i][j] = converted_image[i][j] & 224
    return converted_image


image = plt.imread('landscape.jpg')
gray_image = make_gray_image(image)
converted_image = last_three_bits(gray_image)

fig, ax = plt.subplots(1,2, figsize=(8, 7))
ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(converted_image, cmap='gray')
ax[1].set_title(f'MSB-3 Only')

plt.tight_layout()
plt.show()
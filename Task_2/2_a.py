import numpy as np
import matplotlib.pyplot as plt
def convert_to_gray(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image

def brightness_enhancement(gray_image, factor, low, high):
    [height, width] = gray_image.shape
    for i in range(height):
        for j in range(width):
            if gray_image[i][j] >= low and gray_image[i][j] <= high:
                gray_image[i][j] += 50
                if(gray_image[i][j] > 255):
                    gray_image[i][j] = 255
                if(gray_image[i][j] < 0):
                    gray_image[i][j] = 0
    return gray_image


image = plt.imread('skull.jpg')
gray_image = convert_to_gray(image)
factor, low, high = 50, 105, 200
enhanced_image = brightness_enhancement(gray_image, factor, low, high)

fig, ax = plt.subplots(1,2, figsize=(8, 7))
ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(enhanced_image, cmap='gray')
ax[1].set_title(f'Enhanced between[{low}-{high}] by {factor}')

plt.tight_layout()
plt.show()
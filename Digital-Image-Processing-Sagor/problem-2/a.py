import numpy as np
import matplotlib.pyplot as plt


def make_gray_image(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image

def brightness_enhancement(image, enhancement_factor, min_intencity, max_intencity):
    enhanced_image = np.copy(image)

    for y in range(enhanced_image.shape[0]):
        for x in range(enhanced_image.shape[1]):
            gray_value = enhanced_image[y, x]
            if gray_value >= min_intencity and gray_value <= max_intencity:
                new_gray_value = gray_value + enhancement_factor
                if new_gray_value > 255:
                    new_gray_value = 255
                elif new_gray_value < 0:
                    new_gray_value = 0
                enhanced_image[y, x] = new_gray_value

    return enhanced_image


rgb_image = plt.imread('images/skull.jpg')
gray_image = make_gray_image(rgb_image)

factor, low, high = 50, 150, 205
enhanced_image = brightness_enhancement(gray_image, factor, low, high)


fig, ax = plt.subplots(1,2, figsize=(8, 7))

ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(enhanced_image, cmap='gray')
ax[1].set_title(f'Enhanced between[{low}-{high}] by {factor}')

plt.tight_layout()
plt.show()

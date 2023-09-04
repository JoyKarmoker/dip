import numpy as np
import matplotlib.pyplot as plt


def make_gray_image(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image


def make_last_three(image):
    #10000000->128
    #11000000->192
    #11100000->224
    #11110000->240
    #11111000->248
    #11111100->252
    #11111110->254
    #11111111->255

    mask = 128
    new_image = image.astype(np.uint8) & mask
    return new_image


rgb_image = plt.imread('images/landscape.jpg')
gray_image = make_gray_image(rgb_image)

new_image = make_last_three(gray_image)

plt.figure(figsize=(8, 7))
plt.subplot(121)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')
plt.subplot(122)
plt.imshow(new_image, cmap='gray')
plt.title('MSB-3 only')

plt.tight_layout()
plt.show()

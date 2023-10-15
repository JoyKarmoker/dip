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


def difference_image(original_image, msb_image):
    [height, width] = original_image.shape
    difference_image = np.zeros((int(height), int(width))).astype(np.uint8)
    for i in range(height):
        for j in range(width):
            difference_image[i][j] = original_image[i][j] - msb_image[i][j]
    return difference_image



image = plt.imread('landscape.jpg')
gray_image = make_gray_image(image)
converted_image = last_three_bits(gray_image)
diff_image = difference_image(gray_image, converted_image)

plt.figure(figsize=(8,7))
plt.subplot(2,2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')
plt.subplot(2,2, 2)
plt.imshow(converted_image, cmap='gray')
plt.title('MSB-3 Only')
plt.subplot(2,2,(3,4))
plt.imshow(diff_image, cmap='gray')
plt.title('Difference Image')

plt.tight_layout()
plt.show()
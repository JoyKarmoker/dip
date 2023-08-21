import cv2
import numpy as np
image = cv2.imread('monalisa.png')


def divide_image_blocks(img, n_blocks=(2, 2)):
    horizontal = np.array_split(img, n_blocks[0])
    splitted_img = [np.array_split(block, n_blocks[1], axis=1) for block in horizontal]
    return np.asarray(splitted_img, dtype=np.ndarray)


result = divide_image_blocks(image)
images = []

for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        sub_array = result[i][j]
        sub_array = sub_array.astype(np.uint8)
        images.append(sub_array)

for idx, img in enumerate(images):
    cv2.imshow(f'block_{idx}', img)


# Reassemble the full image
full_image = np.vstack([np.hstack(images[0:2]), np.hstack(images[2:4])])
cv2.imshow('full_image from separated image', full_image)

cv2.waitKey(0)

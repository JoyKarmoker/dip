import numpy as np
import matplotlib.pyplot as plt


def make_gray_image(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image


def make_up_down_sampling(image, sampling_factor, type=0):
    # type=0 -> downsampling
    # type=1 -> upsampling
    if type == 0:
        new_height = int(image.shape[0] // sampling_factor)
        new_width = int(image.shape[1] // sampling_factor)
    else:
        new_height = int(image.shape[0] * sampling_factor)
        new_width = int(image.shape[1] * sampling_factor)

    sampled_image = np.empty((new_height, new_width), dtype=image.dtype)
    for y in range(new_height):
        for x in range(new_width):
            if type == 0:
                original_x = int(x * sampling_factor)
                original_y = int(y * sampling_factor)
            else:
                original_x = int(x // sampling_factor)
                original_y = int(y // sampling_factor)

            sampled_image[y, x] = image[original_y, original_x]

    return sampled_image


rgb_image = plt.imread('images/cat.jpg')
gray_image = make_gray_image(rgb_image)

sampled_image_array = []
sampled_image = gray_image
type = 0
for i in range(8):
    sampled_image_array.append(sampled_image)
    # if i == 3:
    #     type = 1
    sampled_image = make_up_down_sampling(sampled_image, 2, type)
    print(f'OK-{i+1}')



row, col = 2, 4
fig, ax = plt.subplots(row, col, figsize=(9, 7))

idx = 0
for i in range(row):
    for j in range(col):
        ax[i, j].imshow(sampled_image_array[idx], cmap='gray')
        h = sampled_image_array[idx].shape[0]
        w = sampled_image_array[idx].shape[1]
        ax[i, j].set_title(f'{h}x{w}')
        idx += 1

plt.tight_layout()
plt.show()

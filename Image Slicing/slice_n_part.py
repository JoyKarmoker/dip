import cv2
import numpy as np
image = cv2.imread('monalisa.png')
# cv2.imshow('Full Image', image)

def resize_image_to_divisible(image, m, n):
    height, width, channels = image.shape
    new_height = height + (m - (height % m)) % m
    new_width = width + (n - (width % n)) % n
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


def divide_image_blocks(img, row=3, column=3):
    horizontal = np.array_split(img, row)
    splitted_image = [np.array_split(blocks, column, axis=1) for blocks in horizontal]
    return np.asarray(splitted_image, dtype=np.ndarray)


row = int(input('How many rows you want to divide: '))
column = int(input('How many columns you want to divide: '))

image = resize_image_to_divisible(image, row, column)
result = divide_image_blocks(image, row, column)
image_slices = []

for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        sub_array = result[i][j]
        sub_array = sub_array.astype(np.uint8)
        image_slices.append(sub_array)

for idx, img in enumerate(image_slices):
    cv2.imshow(f'block{idx}', img)

# Reassemble the full image
# Calculate the number of rows and columns
num_rows = row
num_cols = column

# Initialize an empty list to store the rows of the full image
rows = []
#
# Iterate through the image_slices list to combine the rows of slices
for i in range(0, len(image_slices), num_cols):
    row_slice = image_slices[i:i + num_cols]
    row_image = np.hstack(row_slice)
    rows.append(row_image)

# # Combine the rows to create the full image
full_image = np.vstack(rows)

# Display the full image
cv2.imshow('Reassembled Full Image', full_image)

cv2.waitKey(0)

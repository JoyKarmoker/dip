import numpy as np
import cv2
image = cv2.imread('monalisa.png')


def resize_image_to_divisible(image, row, col):
    height, width, channel = image.shape
    new_height = height + (col - (height % col)) % col
    new_width = width + (row - (width % row)) % row
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


def divide_image_blocks(image, row=3, col=3):
    horizzontal = np.array_split(image, row)
    splitted_image = [np.array_split(block, col, axis=1) for block in horizzontal]
    return np.asarray(splitted_image, dtype=np.ndarray)


row = int(input('How many rows do you want to slice: '))
column = int(input('How many collumns do you want to slice: '))

image = resize_image_to_divisible(image, row, column)
result = divide_image_blocks(image, row, column)
print(result.shape)
image_slices = []

for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        sub_array = result[i][j]
        sub_array = sub_array.astype(np.uint8)
        image_slices.append(sub_array)

for idx, img in enumerate(image_slices):
    cv2.imshow(f'block{idx}', img)


#reassemble full image
num_rows = row
num_columns = column
rows = []

for i in range(0, len(image_slices), num_columns):
    row_slice = image_slices[i:i+num_columns]
    row_image = np.hstack(row_slice)
    rows.append(row_image)

full_image = np.vstack(rows)
cv2.imshow('Full Image from sliced image', full_image)


cv2.waitKey(0)

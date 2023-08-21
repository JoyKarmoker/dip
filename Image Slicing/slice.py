import cv2
import numpy as np
image = cv2.imread('monalisa.png')
print(image.shape)
#number_of_slice = input('How many Slices do you want? ')
cv2.imshow('Full Image Before Slicing', image)
height, width, channel = image.shape
half_height = height//2
#top_image = image[:half_height, :]
#bottom_image = image[half_height:, :]
#top_image = image[:half_height]
#bottom_image = image[half_height:]
top_image = image[:half_height, :, :]
bottom_image = image[half_height:, :, :]

full_image = np.concatenate((top_image, bottom_image), axis=0)
cv2.imshow('Top Image', top_image)
cv2.imshow('Bottom Image', bottom_image)
cv2.imshow('Full Image after adding', full_image)
print(full_image.shape)
print(full_image.shape[0])
cv2.waitKey(0)



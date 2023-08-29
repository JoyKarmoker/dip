import cv2
import numpy as np


img1 = cv2.imread('monalisa.png', 0)

[m, n] = img1.shape
print('Image Shape:', m, n)

cv2.namedWindow('Downsampling', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Downsampling', 512, 512)


print('Original Image:')
cv2.imshow('Downsampling', img1)


# Initialize the down-sampling rate
f = 2

while True:
    # Wait for a key press
    key = cv2.waitKey(0)

    if key == ord('q') or key == 27:  # 'q' key or ESC key
        break

    # Downsample the imageq
    img2 = np.zeros((m // f, n // f), dtype=np.uint8)

    if img2.shape[0] > 0 and img2.shape[1] > 0:  # Check for valid index
        for i in range(0, m, f):
            for j in range(0, n, f):
                try:
                    img2[i // f][j // f] = img1[i][j]
                except IndexError:
                    pass

        #print('Down Sampled Image:')
        cv2.imshow('Downsampling', img2)

    # Increase the down-sampling rate when any other key is pressed
    f *= 2


cv2.destroyAllWindows()

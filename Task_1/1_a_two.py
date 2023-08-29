import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the original image
img1 = cv2.imread('monalisa.png', 0)

# Obtain the size of the original image
[m, n] = img1.shape
print('Image Shape:', m, n)

# Create a window for displaying images
cv2.namedWindow('Downsampling', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Downsampling', 512, 512)

# Initialize the down-sampling rate
f = 2

while True:
    # Show the original image
    print('Original Image:')
    plt.imshow(img1, cmap="gray")
    plt.show()

    # Show the downsampled image
    print('Down Sampled Image:')
    img2 = np.zeros((m // f, n // f), dtype=int)
    for i in range(0, m, f):
        for j in range(0, n, f):
            try:
                img2[i // f][j // f] = img1[i][j]
            except IndexError:
                pass
    plt.imshow(img2, cmap="gray")
    plt.show()

    # Wait for a key press
    key = cv2.waitKey(0)

    # Check if the 'q' key (quit) is pressed
    if key == ord('q') or key == 27:  # 'q' key or ESC key
        break

    # Increase the down-sampling rate when any other key is pressed
    f *= 2

# Close the OpenCV window
cv2.destroyAllWindows()

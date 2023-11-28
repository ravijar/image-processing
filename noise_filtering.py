# the input image should be renamed as 'noisy_image.jpg'

import numpy as np
import cv2

# mean filter implementation
def mean_filter(image, kernel_size):
    pad_size = kernel_size // 2
    result = np.zeros_like(image, dtype=np.uint8)

    for i in range(pad_size, image.shape[0] - pad_size):
        for j in range(pad_size, image.shape[1] - pad_size):
            for c in range(image.shape[2]):  # loop over color channels
                result[i, j, c] = np.mean(image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1, c])

    return result

# median filter implementation
def median_filter(image, kernel_size):
    pad_size = kernel_size // 2
    result = np.zeros_like(image, dtype=np.uint8)

    for i in range(pad_size, image.shape[0] - pad_size):
        for j in range(pad_size, image.shape[1] - pad_size):
            for c in range(image.shape[2]):  # loop over color channels
                result[i, j, c] = np.median(image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1, c])

    return result

# k closest averaging filter implementation
def k_closest_averaging(image, kernel_size, k):
    pad_size = kernel_size // 2
    result = np.zeros_like(image, dtype=np.uint8)

    for i in range(pad_size, image.shape[0] - pad_size):
        for j in range(pad_size, image.shape[1] - pad_size):
            for c in range(image.shape[2]):  # loop over color channels
                neighbors = image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1, c]
                sorted_neighbors = np.sort(neighbors.flatten())
                result[i, j, c] = np.mean(sorted_neighbors[:k])

    return result

# threshold averaging filter implementation
def threshold_averaging(image, kernel_size, threshold):
    pad_size = kernel_size // 2
    result = np.zeros_like(image, dtype=np.uint8)

    for i in range(pad_size, image.shape[0] - pad_size):
        for j in range(pad_size, image.shape[1] - pad_size):
            for c in range(image.shape[2]):  # loop over color channels
                neighbors = image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1, c]
                result[i, j, c] = np.mean(neighbors) if np.mean(neighbors) > threshold else image[i, j, c]

    return result


# reading image
image = cv2.imread('noisy_image.jpg')

# apply mean filter
result_mean = mean_filter(image, kernel_size=3)

# apply median filter
result_median = median_filter(image, kernel_size=3)

# apply k-closest averaging
result_k_closest = k_closest_averaging(image, kernel_size=3, k=5)

# apply threshold averaging
result_threshold = threshold_averaging(image, kernel_size=3, threshold=50)

# display results
cv2.imshow('Original Image', image)
cv2.imshow('Mean Filter Result', result_mean)
cv2.imshow('Median Filter Result', result_median)
cv2.imshow('K-Closest Averaging Result', result_k_closest)
cv2.imshow('Threshold Averaging Result', result_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()


# the input image must be in the root folder named as "image.jpg"
# change the scaling factor accordingly

import cv2
import numpy


def get_nearest_neighbour(i, j, root_i, root_j):
    # returns the nearest neighbour
    if i-root_i > 0.5:
        if j-root_j > 0.5:
            return [root_i+1, root_j+1]
        else:
            return [root_i+1, root_j]
    else:
        if j-root_j > 0.5:
            return [root_i, root_j+1]
        else:
            return [root_i, root_j]


def scale_nearest_neighbour(original_image, scaling_factor):
    # taking original image parameters
    original_image_height, original_image_width, original_image_colors = original_image.shape
    # calculating output image parameters
    output_image_height = numpy.floor(
        original_image_height * scaling_factor).astype(int)
    output_image_width = numpy.floor(
        original_image_width * scaling_factor).astype(int)
    output_image = numpy.zeros(
        (output_image_height, output_image_width, original_image_colors), dtype=numpy.uint8)

    for i in range(output_image_height):
        # taking x coordinate relevant to original image
        original_i = int(i/scaling_factor)
        for j in range(output_image_width):
            # taking y coordinate relevant to original image
            original_j = int(j/scaling_factor)
            # taking nearest neighbour coordinates
            nearest_i, nearest_j = get_nearest_neighbour(
                i/scaling_factor, j/scaling_factor, original_i, original_j)

            # correcting out of boudry occasions
            if nearest_i >= original_image_height:
                nearest_i = original_image_height - 1
            if nearest_j >= original_image_width:
                nearest_j = original_image_width - 1

            # looping for all color channels
            for k in range(original_image_colors):
                output_image[i, j, k] = original_image[nearest_i, nearest_j, k]

    return output_image


def scale_bi_linear_interpolation(original_image, scaling_factor):
    # taking original image parameters
    original_image_height, original_image_width, original_image_colors = original_image.shape
    # calculating output image parameters
    output_image_height = numpy.floor(
        original_image_height * scaling_factor).astype(int)
    output_image_width = numpy.floor(
        original_image_width * scaling_factor).astype(int)
    output_image = numpy.zeros(
        (output_image_height, output_image_width, original_image_colors), dtype=numpy.uint8)

    for i in range(output_image_height):
        # taking x coordinate relevant to original image
        original_i = int(i/scaling_factor)
        for j in range(output_image_width):
            # taking y coordinate relevant to original image
            original_j = int(j/scaling_factor)

            # taking distances to scaled pixel from root pixel
            delta_i = i/scaling_factor - original_i
            delta_j = j/scaling_factor - original_j

            # looping for all color channels
            for k in range(original_image_colors):

                # correcting out of boudry occasions
                original_i_next = original_i + 1 if original_i + \
                    1 < original_image_height else original_image_height - 1
                original_j_next = original_j + 1 if original_j + \
                    1 < original_image_width else original_image_width - 1

                # calculating interpolated value
                interpolated_value = int(original_image[original_i, original_j, k]*delta_i*delta_j + original_image[original_i_next, original_j, k]*(
                    1-delta_i)*delta_j + original_image[original_i, original_j_next, k]*delta_i*(1-delta_j) + original_image[original_i_next, original_j_next, k]*(1-delta_i)*(1-delta_j))

                output_image[i, j, k] = interpolated_value

    return output_image


# reading the image
path = "./image.jpg"
image = cv2.imread(path)
scaling_factor = 3

nearest_neighbour = scale_nearest_neighbour(image,scaling_factor)
bi_linear_interpolation = scale_bi_linear_interpolation(image,scaling_factor)

# displaying the image
cv2.imshow('Original', image)
cv2.imshow('Scale Nearest Neighbour', nearest_neighbour)
cv2.imshow('Scale Bi Linear Interpolation', bi_linear_interpolation)

cv2.waitKey(0)

# saving the image
cv2.imwrite('Scale Nearest Neighbour.jpg', nearest_neighbour)
cv2.imwrite('Scale Bi Linear Interpolation.jpg', bi_linear_interpolation)

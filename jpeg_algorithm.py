import cv2
import numpy as np
import json
import pickle


def centralize(block, value):
    centralized_block = block.astype(np.float32) - value
    return centralized_block


def apply_DCT(block):
    output_block = np.empty_like(block)
    N = len(block)
    for u in range(N):
        for v in range(N):
            sum = 0.0
            for x in range(N):
                for y in range(N):
                    sum += block[x][y]*np.cos(((2*x+1)*u*np.pi)/(2*N)) * \
                        np.cos(((2*y+1)*v*np.pi)/(2*N))
            if u == 0:
                cu = 1/np.sqrt(2)
            else:
                cu = 1
            if v == 0:
                cv = 1/np.sqrt(2)
            else:
                cv = 1
            output_block[u][v] = round(0.25*cu*cv*sum, 1)
    return output_block


def quantize(block, quantization_table):
    quantized_block = np.round(block/quantization_table).astype(np.int32)
    return quantized_block


def zigzag_scan(block):
    rows, cols = len(block), len(block[0])
    result = []

    for i in range(rows + cols - 1):
        if i % 2 == 0:  # Even diagonal (up-right)
            if i < rows:
                row, col = i, 0
            else:
                row, col = rows - 1, i - rows + 1
            while row >= 0 and col < cols:
                result.append(block[row][col])
                row -= 1
                col += 1
        else:  # Odd diagonal (down-left)
            if i < cols:
                row, col = 0, i
            else:
                row, col = i - cols + 1, cols - 1
            while row < rows and col >= 0:
                result.append(block[row][col])
                row += 1
                col -= 1

    return result


def run_length_encode(arr):
    if not arr:
        return []

    encoded_array = []
    current_element = arr[0]
    count = 1

    for element in arr[1:]:
        if element == current_element:
            count += 1
        else:
            encoded_array.append((int(current_element), count))
            current_element = element
            count = 1

    encoded_array.append((int(current_element), count))
    return encoded_array


quantization_table_Y = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float32)

# Read the image
path = "./image.jpg"
image = cv2.imread(path)

# Convert the image to YCrCb
image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Extract only the Y channel for convinience
image = image[:, :, 0]

height, width = image.shape[:2]
block_size = 8

# Calculate the number of blocks in each dimension
num_blocks_x = (width + block_size - 1) // block_size
num_blocks_y = (height + block_size - 1) // block_size

# Zero-padded canvas to hold the blocks
canvas = np.zeros((num_blocks_y * block_size, num_blocks_x *
                  block_size), dtype=np.uint8)

# NumPy array to store the blocks
blocks_matrix = np.zeros(
    (num_blocks_y, num_blocks_x, block_size, block_size), dtype=np.uint8)

# Copy the image onto the canvas
canvas[:height, :width] = image

# Iterate through the canvas and create 8x8 blocks
for y in range(0, canvas.shape[0], block_size):
    for x in range(0, canvas.shape[1], block_size):
        # Extract an 8x8 block
        block = canvas[y:y+block_size, x:x+block_size]

        # Calculate the block's position in the 3D array
        block_x = x // block_size
        block_y = y // block_size

        # Store the block in the 3D array
        blocks_matrix[block_y, block_x] = block

encoded_data = [[] for _ in range(num_blocks_y)]

for y in range(blocks_matrix.shape[0]):
    for x in range(blocks_matrix.shape[1]):
        blocks_matrix[y, x] = centralize(blocks_matrix[y, x], 128)
        blocks_matrix[y, x] = apply_DCT(blocks_matrix[y, x])
        blocks_matrix[y, x] = quantize(
            blocks_matrix[y, x], quantization_table_Y)
        encoded_data[y].append(
            run_length_encode(zigzag_scan(blocks_matrix[y, x])))


# Display the original image
cv2.imshow('Original', image)
cv2.waitKey(0)

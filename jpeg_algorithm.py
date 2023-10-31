import cv2
import numpy as np


def apply_DCT(block):
    DCT_block = np.empty_like(block, dtype=np.float32)
    N = len(block)

    # applying DCT for a block
    for u in range(N):
        for v in range(N):
            DCT_val = 0.0
            for x in range(N):
                for y in range(N):
                    DCT_val += block[x][y] * \
                        np.cos((2*x+1)*u*np.pi/(2*N)) * \
                        np.cos((2*y+1)*v*np.pi/(2*N))
            DCT_block[u][v] = DCT_val

    return DCT_block


# Read the image
path = "./image.jpg"
image = cv2.imread(path)

# Convert the image to YCrCb
image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Extract only the Y channel for convinience
image = image[:,:,0]

height, width = image.shape[:2]
block_size = 8

# Calculate the number of blocks in each dimension
num_blocks_x = (width + block_size - 1) // block_size
num_blocks_y = (height + block_size - 1) // block_size

# Zero-padded canvas to hold the blocks
canvas = np.zeros((num_blocks_y * block_size, num_blocks_x *
                  block_size), dtype=np.uint8)

# 3D NumPy array to store the blocks
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


for y in range(blocks_matrix.shape[0]):
    for x in range(blocks_matrix.shape[1]):
        blocks_matrix[y, x] = apply_DCT(blocks_matrix[y, x])



# Display the original image
cv2.imwrite('block.jpg', blocks_matrix[0, 50])
cv2.imshow('Original', image)
cv2.waitKey(0)

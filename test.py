import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def hypothetical_reconstructed_reference(width, height):
    return np.full((height, width), 128, dtype=np.uint8)

def mean_absolute_error(block1, block2):
    """Calculate Mean Absolute Error (MAE) between two blocks."""
    return np.mean(np.abs(block1.astype(np.int16) - block2.astype(np.int16)))

def full_search(current_block, ref_frame, block_position, block_size, search_range):
    """Perform full search for the best matching block."""
    best_mae = float('inf')
    best_motion_vector = (0, 0)
    y, x = block_position

    # Define the search window (bounded within the frame)
    y_start = max(0, y - search_range)
    y_end = min(ref_frame.shape[0] - block_size, y + search_range)
    x_start = max(0, x - search_range)
    x_end = min(ref_frame.shape[1] - block_size, x + search_range)

    # Iterate over the search range
    for dy in range(y_start, y_end + 1):
        for dx in range(x_start, x_end + 1):
            ref_block = ref_frame[dy:dy + block_size, dx:dx + block_size]
            mae = mean_absolute_error(current_block, ref_block)

            # Tie-breaking: prioritize smaller motion vectors
            motion_vector = (dy - y, dx - x)
            if mae < best_mae or (mae == best_mae and (abs(motion_vector[0]) + abs(motion_vector[1])) < (abs(best_motion_vector[0]) + abs(best_motion_vector[1]))):
                best_mae = mae
                best_motion_vector = motion_vector

    return best_mae, best_motion_vector

def round_to_nearest_multiple(block, n):
    """Round each element in the block to the nearest multiple of 2^n."""
    factor = 2 ** n
    return np.round(block / factor) * factor

def calculate_residual(current_block, predicted_block):
    """Calculate the residual block by subtracting the predicted block from the current block."""
    return current_block.astype(np.int16) - predicted_block.astype(np.int16)

def reconstruct_block(predicted_block, approximated_residual):
    """Reconstruct the block by adding the approximated residual to the predicted block."""
    return np.clip(predicted_block + approximated_residual, 0, 255).astype(np.uint8)

def process_frame_and_reconstruct(Y_current, ref_frame, block_size, search_range, motion_vectors, n):
    """Process each block in the current frame and reconstruct the Y-frame."""
    # Split the current frame into blocks
    blocks = split_into_blocks(Y_current, block_size)

    total_mae = 0
    num_blocks = len(blocks)

    reconstructed_frame = np.zeros_like(Y_current)
    residual_frame = np.zeros_like(Y_current)
    approx_residual_frame = np.zeros_like(Y_current)

    # Process each block and reconstruct
    for current_block, block_position in blocks:
        mae, motion_vector = full_search(current_block, ref_frame, block_position, block_size, search_range)
        
        # Calculate the predicted block based on motion vector
        pred_y = block_position[0] + motion_vector[0]
        pred_x = block_position[1] + motion_vector[1]

        if 0 <= pred_y < ref_frame.shape[0] - block_size and 0 <= pred_x < ref_frame.shape[1] - block_size:
            predicted_block = ref_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
        else:
            predicted_block = np.zeros((block_size, block_size), dtype=np.uint8)  # Default block if out of bounds
        
        # Calculate residual block and approximate it
        residual_block = calculate_residual(current_block, predicted_block)
        approximated_residual = round_to_nearest_multiple(residual_block, n)

        # Reconstruct the block
        reconstructed_block = reconstruct_block(predicted_block, approximated_residual)

        # Place the reconstructed block and residual back into the respective frames
        reconstructed_frame[block_position[0]:block_position[0] + block_size, block_position[1]:block_position[1] + block_size] = reconstructed_block
        residual_frame[block_position[0]:block_position[0] + block_size, block_position[1]:block_position[1] + block_size] = residual_block
        approx_residual_frame[block_position[0]:block_position[0] + block_size, block_position[1]:block_position[1] + block_size] = approximated_residual

    return reconstructed_frame, residual_frame, approx_residual_frame

def save_y_frame_to_file(frame, filename):
    """Save the reconstructed Y frame to a file."""
    np.savetxt(filename, frame, fmt='%d')

def compare_frames(original_frame, reconstructed_frame):
    """Compare two Y frames using PSNR and SSIM."""
    psnr_value = peak_signal_noise_ratio(original_frame, reconstructed_frame)
    ssim_value = structural_similarity(original_frame, reconstructed_frame)
    return psnr_value, ssim_value

def visualize_frames(original_frame, reference_frame, predicted_frame, residual_frame, approx_residual_frame, reconstructed_frame):
    """Visualize all frames using matplotlib."""
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.title('Source (Current Frame)')
    plt.imshow(original_frame, cmap='gray')
    
    plt.subplot(2, 3, 2)
    plt.title('Reference (Previous Frame)')
    plt.imshow(reference_frame, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Predicted Frame')
    plt.imshow(predicted_frame, cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title('Residual Frame')
    plt.imshow(residual_frame, cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title('Approximated Residual Frame')
    plt.imshow(approx_residual_frame, cmap='gray')

    plt.subplot(2, 3, 6)
    plt.title('Reconstructed Frame')
    plt.imshow(reconstructed_frame, cmap='gray')

    plt.tight_layout()
    plt.show()

def process_and_compare_all_frames(y_files, block_size, search_range, n):
    frame_counter = 0

    for file_path in y_files:
        Y_current = load_y_component(file_path)

        if frame_counter == 0:
            Y_reference = hypothetical_reconstructed_reference(Y_current.shape[1], Y_current.shape[0])
        else:
            Y_reference = load_y_component(y_files[frame_counter - 1])

        # Reconstruct the current frame and get residuals
        reconstructed_frame, residual_frame, approx_residual_frame = process_frame_and_reconstruct(
            Y_current, Y_reference, block_size, search_range, [], n)

        # Save the reconstructed frame
        save_y_frame_to_file(reconstructed_frame, f'reconstructed_frame_{frame_counter}.txt')

        # Compare the original and reconstructed frames
        psnr_value, ssim_value = compare_frames(Y_current, reconstructed_frame)
        print(f'Frame {frame_counter}: PSNR = {psnr_value}, SSIM = {ssim_value}')

        # Visualize the frames
        predicted_frame = hypothetical_reconstructed_reference(Y_current.shape[1], Y_current.shape[0])  # Placeholder for actual predicted frame
        visualize_frames(Y_current, Y_reference, predicted_frame, residual_frame, approx_residual_frame, reconstructed_frame)

        frame_counter += 1

# Example usage
input_directory = 'y_only_files'
y_files = get_yuv_files(input_directory)
block_size = 8
search_range = 4
n = 2  # Set rounding to nearest multiple of 2^n

process_and_compare_all_frames(y_files, block_size, search_range, n)

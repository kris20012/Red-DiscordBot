import numpy as np
import os

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

def process_frame(Y_current, ref_frame, block_size, search_range, motion_vectors):
    """Process each block in the current frame using full search."""
    # Split the current frame into blocks
    blocks = split_into_blocks(Y_current, block_size)
    
    total_mae = 0
    num_blocks = len(blocks)
    
    # For each block, find the best matching block in the reference frame
    for current_block, block_position in blocks:
        mae, motion_vector = full_search(current_block, ref_frame, block_position, block_size, search_range)
        if mae is not None:
            total_mae += mae
            if mae < float('inf'):  # This checks if a valid motion vector was found
                motion_vectors.append((block_position, motion_vector))  # Store block position and motion vector
        else:
            num_blocks -= 1 
        
        # Calculate the predicted block based on motion vector
        pred_y = block_position[0] + motion_vector[0]
        pred_x = block_position[1] + motion_vector[1]
        
        if 0 <= pred_y < ref_frame.shape[0] - block_size and 0 <= pred_x < ref_frame.shape[1] - block_size:
            predicted_block = ref_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
        else:
            predicted_block = np.zeros((block_size, block_size), dtype=np.uint8)  # Default block if out of bounds
        
        # Calculate residual block
        residual_block = calculate_residual(current_block, predicted_block)
        
        # Round the residual block for n = 1, 2, 3
        for n in range(1, 4):
            approximated_residual = round_to_nearest_multiple(residual_block, n)
            # Here you can store or display the approximated residual as needed
            
        # Store the motion vector with its block position
        motion_vectors.append((block_position[0], block_position[1], motion_vector[0], motion_vector[1]))

    # Calculate average MAE for the frame
    average_mae = total_mae / num_blocks
    return average_mae

def dump_motion_vectors(motion_vectors, filename):
    """Dump motion vectors to a text file."""
    with open(filename, 'w') as f:
        for block_y, block_x, mv_y, mv_x in motion_vectors:
            f.write(f'Block Position: ({block_y}, {block_x}), Motion Vector: ({mv_y}, {mv_x})\n')


input_directory = 'y_only_files'
y_files = get_yuv_files(input_directory)

# Full paths to the YUV files
y_files_full_path = [os.path.join(input_directory, f) for f in y_files]

frame_counter = 0
search_ranges = [1, 4, 8]
block_sizes = [2, 8, 64]

motion_vectors_all = []

for file_path in y_files_full_path:
    Y = load_y_component(file_path)

    if frame_counter == 0:
        Y_reference = hypothetical_reconstructed_reference(width, height)
    else:
        Y_reference = load_y_component(y_files_full_path[frame_counter - 1], width, height)

    for block_size in block_sizes:
        for search_range in search_ranges:
            # Pad the frame if necessary
            if (Y.shape[1] % block_size != 0 or Y.shape[0] % block_size != 0):
                Y_padded = pad_frame(Y, block_size)
            else:
                Y_padded = Y

            motion_vectors = []
            avg_mae = process_frame(Y_padded, Y_reference, block_size, search_range, motion_vectors)
            motion_vectors_all.extend(motion_vectors)
            print(f'Frame {frame_counter + 1}, Block Size {block_size}, Search Range {search_range}, Avg MAE: {avg_mae}')

    frame_counter += 1
    if (frame_counter >= 1):
        break

dump_motion_vectors(motion_vectors_all, 'motion_vectors.txt')

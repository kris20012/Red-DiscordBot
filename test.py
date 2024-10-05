def load_y_component(file_path, height, width):
    """Load the Y component from a Y-only file."""
    Y = np.fromfile(file_path, dtype=np.uint8)
    return Y.reshape((height, width))

def split_into_blocks(Y, block_size):
    """Split Y-component into blocks of size block_size x block_size."""
    height, width = Y.shape
    blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = Y[y:y + block_size, x:x + block_size]
            blocks.append((block, (y, x)))  # Return both block and its position
    return blocks

def pad_frame(Y, block_size):
    """Pad frame if necessary, filling with gray (128)."""
    height, width = Y.shape
    pad_height = (block_size - (height % block_size)) % block_size
    pad_width = (block_size - (width % block_size)) % block_size

    if pad_height > 0 or pad_width > 0:
        Y_padded = np.pad(Y, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=128)
    else:
        Y_padded = Y

    return Y_padded

def mean_absolute_error(block1, block2):
    """Calculate Mean Absolute Error (MAE) between two blocks."""
    return np.mean(np.abs(block1.astype(np.int16) - block2.astype(np.int16)))

def full_search(current_block, ref_frame, block_position, block_size, search_range):
    """Perform full search for the best matching block."""
    best_mae = float('inf')
    best_vector = (0, 0)
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
            if mae < best_mae or (mae == best_mae and (abs(motion_vector[0]) + abs(motion_vector[1])) < (abs(best_vector[0]) + abs(best_vector[1]))):
                best_mae = mae
                best_vector = motion_vector

    return best_mae, best_vector

def process_frame(Y_current, ref_frame, block_size, search_range):
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
        else:
            num_blocks -= 1  
    
    # Calculate average MAE for the frame
    average_mae = total_mae / num_blocks
    print(f'Frame {frame_counter + 1},')
    return average_mae

def get_yuv_files(input_dir):
    """Get all YUV files from the directory."""
    return [f for f in os.listdir(input_dir) if f.endswith('.yuv')]

# Main processing loop
input_directory = 'y_only_files'
y_files = get_yuv_files(input_directory)

# Assume the frame size is known
height, width = 288, 352  # Replace with actual dimensions
block_sizes = [2, 8, 64]
search_ranges = [1, 4, 8]

# Full paths to the YUV files
y_files_full_path = [os.path.join(input_directory, f) for f in y_files]

# Process each frame (up to 5 frames)
frame_counter = 0

for file_path in y_files_full_path:
    Y_current = load_y_component(file_path, height, width)
    
    # Assume a hypothetical reference frame filled with 128 for the first frame
    ref_frame = np.full((height, width), 128, dtype=np.uint8)
    
    for block_size in block_sizes:
        for search_range in search_ranges:
            # Pad the current frame if necessary
            if (Y_current.shape[1] % block_size != 0 or Y_current.shape[0] % block_size != 0):
                Y_padded = pad_frame(Y_current, block_size)
            else:
                Y_padded = Y_current

            # Perform full search for this block size and search range
            avg_mae = process_frame(Y_padded, ref_frame, block_size, search_range)
            print(f'Frame {frame_counter + 1}, Block Size {block_size}, Search Range {search_range}, Avg MAE: {avg_mae}')

    # For the next iteration, set the current frame as the reference frame
    ref_frame = Y_current
    
    frame_counter += 1
    if frame_counter >= 1:
        break
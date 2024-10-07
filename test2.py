def display_frame(RGB_frame, frame_num):
    # Display the frame using matplotlib
    plt.figure()
    plt.imshow(RGB_frame)
    plt.title(f'Frame {frame_num}')
    plt.axis('off')  # Turn off axis labels
    plt.show()

def hypothetical_reconstructed_reference(width, height):
    return np.full((height, width), 128, dtype=np.uint8)

def mean_absolute_error(block1, block2):
    return np.mean(np.abs(block1.astype(np.int16) - block2.astype(np.int16)))

def full_search(current_block, ref_frame, block_position, block_size, search_range):
    best_mae = float('inf')
    best_motion_vector = (0, 0)
    y, x = block_position

    y_start = max(0, y - search_range)
    y_end = min(ref_frame.shape[0] - block_size, y + search_range)
    x_start = max(0, x - search_range)
    x_end = min(ref_frame.shape[1] - block_size, x + search_range)

    for dy in range(y_start, y_end + 1):
        for dx in range(x_start, x_end + 1):
            ref_block = ref_frame[dy:dy + block_size, dx:dx + block_size]
            mae = mean_absolute_error(current_block, ref_block)
            motion_vector = (dy - y, dx - x)
            if mae < best_mae or (mae == best_mae and (abs(motion_vector[0]) + abs(motion_vector[1])) < (abs(best_motion_vector[0]) + abs(best_motion_vector[1]))):
                best_mae = mae
                best_motion_vector = motion_vector

    return best_mae, best_motion_vector

def round_to_nearest_multiple(block, n):
    factor = 2 ** n
    return np.round(block / factor) * factor

def calculate_residual(current_block, predicted_block):
    return current_block.astype(np.int16) - predicted_block.astype(np.int16)

def reconstruct_block(predicted_block, approximated_residual):
    return predicted_block + approximated_residual

def process_frame(Y_current, ref_frame, block_size, search_range):
    height, width = Y_current.shape
    reconstructed_frame = np.zeros_like(Y_current)
    predicted_frame = np.zeros_like(Y_current)
    residual_frame = np.zeros_like(Y_current, dtype=np.int16)
    total_mae = 0
    num_blocks = 0

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            current_block = Y_current[y:y + block_size, x:x + block_size]
            mae, motion_vector = full_search(current_block, ref_frame, (y, x), block_size, search_range)

            pred_y = y + motion_vector[0]
            pred_x = x + motion_vector[1]
            if 0 <= pred_y < ref_frame.shape[0] - block_size and 0 <= pred_x < ref_frame.shape[1] - block_size:
                predicted_block = ref_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
            else:
                predicted_block = np.zeros((block_size, block_size), dtype=np.uint8)

            residual_block = calculate_residual(current_block, predicted_block)
            approximated_residual = round_to_nearest_multiple(residual_block, 1)  # Assume n=1 for simplicity

            reconstructed_block = reconstruct_block(predicted_block, approximated_residual)
            reconstructed_frame[y:y + block_size, x:x + block_size] = reconstructed_block

            # Update frames with block values
            predicted_frame[y:y + block_size, x:x + block_size] = predicted_block
            residual_frame[y:y + block_size, x:x + block_size] = residual_block
            
            num_blocks += 1
            total_mae += mae

    average_mae = total_mae / num_blocks
    return reconstructed_frame, predicted_frame, residual_frame, average_mae

def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))

def plot_frames(frames, titles, cmap='gray'):
    fig, axs = plt.subplots(1, len(frames), figsize=(20, 5))
    for i, frame in enumerate(frames):
        axs[i].imshow(frame, cmap=cmap)
        axs[i].set_title(titles[i])
        axs[i].axis('off')
    plt.show()

def process_video(input_directory, height, width, block_size, search_range):
    frame_counter = 0
    y_files = get_yuv_files(input_directory)
    y_files_full_path = [os.path.join(input_directory, f) for f in y_files]
    reconstructed_frames = []
    
    for file_path in y_files_full_path:
        Y_current = load_y_component(file_path)
        # print(f'Filepath: {file_path}, Height: {height}, Width: {width}')
        # display_frame(Y_current, 0)

        if frame_counter == 0:
            Y_reference = hypothetical_reconstructed_reference(width, height)
        else:
            Y_reference = reconstructed_frames[-1]

        reconstructed_frame, predicted_frame, residual_frame, avg_mae = process_frame(Y_current, Y_reference, block_size, search_range)
        print(f'Frame {frame_counter + 1}, Avg MAE: {avg_mae}')

        psnr_value = psnr(Y_current, reconstructed_frame)
        print(f'Frame {frame_counter + 1}, PSNR: {psnr_value}')

        plot_frames(
            [Y_current, Y_reference, predicted_frame, reconstructed_frame, residual_frame],
            ['Source (Y)', 'Reference', 'Predicted', 'Reconstructed', 'Residual']
        )

        reconstructed_frames.append(reconstructed_frame)
        frame_counter += 1
        if frame_counter >= 5:  # Limit to 2 frames for testing
            break

input_directory = 'y_only_files'
block_size = 8
search_range = 4
process_video(input_directory, height, width, block_size, search_range)
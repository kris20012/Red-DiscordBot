def load_motion_vectors(mv_file):
    motion_vectors = {}
    with open(mv_file, 'r') as f:
        for line in f:
            if line.startswith("Block Position"):
                pos_str = line.split(":")[1].strip().replace("(", "").replace(")", "").split(",")
                pos = (int(pos_str[0]), int(pos_str[1]))
            elif line.startswith("Motion Vector"):
                mv_str = line.split(":")[1].strip().replace("(", "").replace(")", "").split(",")
                mv = (int(mv_str[0]), int(mv_str[1]))
                motion_vectors[pos] = mv
    return motion_vectors

def load_residuals(residual_file):
    residuals = {}
    current_block = None
    current_residual = []
    
    with open(residual_file, 'r') as f:
        for line in f:
            if line.startswith("Block Position"):
                if current_block is not None:
                    residuals[current_block] = np.array(current_residual, dtype=np.int16)
                pos_str = line.split(":")[1].strip().replace("(", "").replace(")", "").split(",")
                current_block = (int(pos_str[0]), int(pos_str[1]))
                current_residual = []
            elif line.strip() and (line[0].isdigit() or line[0] == '-'):
                current_residual.append(list(map(lambda x: int(float(x)), line.strip().split())))
                
        if current_block is not None:
            residuals[current_block] = np.array(current_residual, dtype=np.int16)
    
    return residuals

def decode_frame(ref_frame, motion_vectors, residuals, block_size):
    height, width = ref_frame.shape
    reconstructed_frame = np.zeros_like(ref_frame)
    
    for (y, x), mv in motion_vectors.items():
        pred_y = y + mv[0]
        pred_x = x + mv[1]
        
        if 0 <= pred_y < height - block_size and 0 <= pred_x < width - block_size:
            predicted_block = ref_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
        else:
            predicted_block = np.zeros((block_size, block_size), dtype=np.uint8)
        
        residual_block = residuals.get((y, x), np.zeros((block_size, block_size), dtype=np.int16))
        reconstructed_block = predicted_block + residual_block
        reconstructed_block = np.clip(reconstructed_block, 0, 255).astype(np.uint8)
        reconstructed_frame[y:y + block_size, x:x + block_size] = reconstructed_block

    return reconstructed_frame

def process_decoded_video(ref_frame, num_frames, mv_file_prefix, residual_file_prefix, block_size):
    reconstructed_frames = []
    
    for frame_idx in range(num_frames):
        mv_file = f"{mv_file_prefix}_frame_{frame_idx + 1}.txt"
        residual_file = f"{residual_file_prefix}_frame_{frame_idx + 1}.txt"
        
        motion_vectors = load_motion_vectors(mv_file)
        residuals = load_residuals(residual_file)
        
        reconstructed_frame = decode_frame(ref_frame, motion_vectors, residuals, block_size)
        reconstructed_frames.append(reconstructed_frame)
        
        ref_frame = reconstructed_frame
        save_y_frame(reconstructed_frame, f"decoded_Y_frame_{frame_idx + 1}.yuv")
        
    return reconstructed_frames

def save_y_frame(Y_frame, output_file):
    with open(output_file, 'wb') as f:
        Y_frame.tofile(f)

def show_decoded_frames(decoded_frames):
    fig, axs = plt.subplots(1, len(decoded_frames), figsize=(20, 5))
    for i in range(len(decoded_frames)):
        axs[i].imshow(decoded_frames[i], cmap='gray')
        axs[i].set_title(f'Decoded Frame {i + 1}')
        axs[i].axis('off')
    plt.show()

initial_reference_frame = np.full((height, width), 128, dtype=np.uint8)
block_size = 8
num_frames = 5
mv_file_prefix = 'motion_vectors'
residual_file_prefix = 'residuals'

decoded_frames = process_decoded_video(initial_reference_frame, num_frames, mv_file_prefix, residual_file_prefix, block_size)
show_decoded_frames(decoded_frames)

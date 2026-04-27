
import numpy as np
from PIL import Image
from pathlib import Path
import sys

def create_montage(npz_path, out_path, max_frames=30):
    z = np.load(npz_path)
    if 'depth' in z and z['depth'].size > 0:
        data = z['depth'] # (T, H, W)
        is_depth = True
    elif 'rgb' in z and z['rgb'].size > 0:
        data = z['rgb'] # (T, H, W, 3)
        is_depth = False
    else:
        print("No depth or rgb data found")
        return

    t = data.shape[0]
    h, w = data.shape[1], data.shape[2]
    
    indices = np.linspace(0, t - 1, min(t, max_frames), dtype=int)
    n = len(indices)
    
    cols = 6
    rows = (n + cols - 1) // cols
    
    montage = Image.new('RGB', (cols * w, rows * h))
    
    for i, idx in enumerate(indices):
        frame_data = data[idx]
        if is_depth:
            # Normalize depth for visualization (0-255)
            # Assuming depth is in meters, clip to some reasonable range e.g. 0-10m
            d_min, d_max = frame_data.min(), frame_data.max()
            if d_max > d_min:
                frame_norm = ((frame_data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                frame_norm = frame_data.astype(np.uint8)
            frame = Image.fromarray(frame_norm).convert('RGB')
        else:
            frame = Image.fromarray(frame_data)
            
        r, c = divmod(i, cols)
        montage.paste(frame, (c * w, r * h))
        
    montage.save(out_path)
    print(f"Saved montage to {out_path}")

if __name__ == "__main__":
    create_montage(sys.argv[1], sys.argv[2])


import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    npz_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    
    z = np.load(npz_path)
    depth = z['depth'] # (T, H, W)
    contact = z['contact_flag']
    t = depth.shape[0]
    
    # Sample 30 frames
    indices = np.linspace(0, t - 1, 30, dtype=int)
    n = len(indices)
    cols = 6
    rows = (n + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axs = axs.ravel()
    
    for i, idx in enumerate(indices):
        ax = axs[i]
        d = depth[idx]
        
        # Fixed normalization 0-8m for consistent movement perception
        # 0m (near) will be bright/yellow, 8m (far) will be dark/purple
        im = ax.imshow(d, cmap='magma', vmin=0, vmax=8)
        
        title = f"t={idx}"
        if contact[idx]:
            title += " [COLLISION!]"
            # Add a red border for collision frames
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
        
        ax.set_title(title, fontsize=9, color='red' if contact[idx] else 'black')
        ax.axis('off')
        
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
        
    fig.suptitle(f"Dynamic Rollout: {npz_path.name}\n(Fixed 0-8m Range | Red = Collision)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved enhanced colorized depth montage to {out_path}")

if __name__ == "__main__":
    main()

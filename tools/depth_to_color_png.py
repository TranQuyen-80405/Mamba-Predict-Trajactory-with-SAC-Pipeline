
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
    t = depth.shape[0]
    
    indices = np.linspace(0, t - 1, min(t, 30), dtype=int)
    n = len(indices)
    cols = 6
    rows = (n + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axs = axs.ravel()
    
    for i, idx in enumerate(indices):
        ax = axs[i]
        d = depth[idx]
        # Use 'magma' colormap for a "heat" look which is easier to see than grayscale
        im = ax.imshow(d, cmap='magma')
        ax.set_title(f"t={idx}", fontsize=8)
        ax.axis('off')
        
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved colorized depth montage to {out_path}")

if __name__ == "__main__":
    main()

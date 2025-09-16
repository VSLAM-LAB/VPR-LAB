import numpy as np
import matplotlib.pyplot as plt

def show_four_npy(file1, file2, file3, file4, cmap="viridis"):
    # Load the matrices
    m1 = np.load(file1)
    m2 = np.load(file2)
    m3 = np.load(file3)
    m4 = np.load(file4)

    # Plot them
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    mats = [m1, m2, m3, m4]
    titles = [file1, file2, file3, file4]

    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# Example usage:
show_four_npy("/home/alejandro/test0.npy", "/home/alejandro/test0.npy","/home/alejandro/test0.npy","/home/alejandro/test0.npy")

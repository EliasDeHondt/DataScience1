from collections import Iterable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def plot_images(filenames: [str], figsize):
    ncols = len(filenames)
    fig, ax = plt.subplots(1, ncols, figsize=figsize)
    if isinstance(ax, Iterable):
        for i, filename in enumerate(filenames):
            img = mpimg.imread(filename)
            ax[i].imshow(img)
            ax[i].axis('off')
    else:
        img = mpimg.imread(filenames[0])
        ax.imshow(img)
        ax.axis('off')

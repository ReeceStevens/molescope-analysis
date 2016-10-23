from functools import partial
from os import listdir

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD = 110
COLOR_CHANNEL_MAX = 256

def intensity(rgb_val):
    red, green, blue = rgb_val
    return (red + green + blue) / 3


def size_inversion(init_size):
    length, width = init_size
    return (width, length)


def array_from_jpeg(filepath):
    img = Image.open(filepath)
    bw_data = list(map(intensity, list(img.getdata())))
    return np.resize(bw_data, size_inversion(img.size))


def mask_single(threshold, value):
    if value < threshold:
        return 0
    else:
        return value


def mask(image_array, threshold):
    apply_mask = partial(mask_single, threshold)
    return [list(map(apply_mask, row)) for row in image_array]


def image_avg_intensity(image_array):
    masked_image = mask(image_array, EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD)
    nonzero_count = 0
    for row in masked_image:
        for val in row:
            if val != 0:
                nonzero_count += 1
    return np.sum(masked_image) / nonzero_count


def plot_all_datasets():
    all_imagepaths = listdir('./datasets')
    rowlen = 5
    f, axarr = plt.subplots(2,rowlen)
    i = 0
    j = 0
    for path in all_imagepaths:
        if j == rowlen:
            j  = 0
            i += 1
        fullpath = './datasets/' + path
        data = array_from_jpeg(fullpath)
        axarr[i,j].imshow(mask(data, EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD))
        j += 1
    plt.show()


def color_histogram(filepath):
    img = Image.open(filepath)
    color_data = img.getdata()
    length, width = size_inversion(img.size)
    red, green, blue = [list(color) for color in zip(*color_data)]
    plt.hist(blue, COLOR_CHANNEL_MAX)
    plt.hold(True)
    plt.hist(green, COLOR_CHANNEL_MAX)
    plt.hist(red, COLOR_CHANNEL_MAX)
    plt.xlabel('Color Intensity')
    plt.ylabel('Number of Pixels')
    plt.title('Color Histogram')
    plt.show()


def main():
    # plot_all_datasets()
    color_histogram('./datasets/IMG_0362.JPG')


main()

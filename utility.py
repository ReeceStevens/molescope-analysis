from os import listdir
from functools import partial

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD = 55
COLOR_CHANNEL_MAX = 256

def intensity(rgb_val):
    red, green, blue = rgb_val
    return (red + green + blue) / 3


def size_inversion(init_size):
    length, width = init_size
    return (width, length)


def color_size_conversion(init_size):
    length, width = init_size
    return (width, length, 3)


def flatten(data_matrix):
    flattened = []
    for row in data_matrix:
        flattened.extend(row)
    return flattened


def array_from_jpeg(filepath):
    img = Image.open(filepath)
    bw_data = list(map(intensity, list(img.getdata())))
    return np.resize(bw_data, size_inversion(img.size))


def mask_single(threshold, value):
    if value < threshold:
        return 0
    else:
        return value


def color_mask_single(threshold, value):
    if intensity(value) < threshold:
        return (0,0,0)
    else:
        red, green, blue = value
        return (COLOR_CHANNEL_MAX-red, COLOR_CHANNEL_MAX-green, COLOR_CHANNEL_MAX-blue)


def mask(image_array, threshold):
    apply_mask = partial(mask_single, threshold)
    return [list(map(apply_mask, row)) for row in image_array]


def color_mask(image_array, threshold):
    apply_mask = partial(color_mask_single, threshold)
    return list(map(apply_mask, image_array))


def color_histogram_diff(figure, color1, color2):
    vals1, bins1 = np.histogram(color1, bins=COLOR_CHANNEL_MAX, range=(0,COLOR_CHANNEL_MAX))
    vals2, bins2 = np.histogram(color2, bins=COLOR_CHANNEL_MAX, range=(0,COLOR_CHANNEL_MAX))
    # We ignore the zero bucket since the mask will blow this up
    vals1[0] = 0
    vals2[0] = 0
    diff_vals = list(map(abs, list(np.subtract(vals2, vals1))))
    figure.bar(bins1[1:], diff_vals, alpha=0.7)
    return diff_vals, bins1


def get_rgb_from_masked_color_array(color_array):
    masked_color_vals_only = [val for val in color_array if val != (0,0,0)]
    red, green, blue = [list(color) for color in zip(*masked_color_vals_only)]
    return red, green, blue


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



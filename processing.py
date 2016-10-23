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


def single_image_analysis(filepath):
    img = Image.open(filepath)
    color_data = img.getdata()
    bw_data = list(map(intensity, list(color_data)))
    img_data = np.resize(bw_data, size_inversion(img.size))

    f, axarr = plt.subplots(2,2)
    f.suptitle('Single Image Analysis for ' + filepath)
    axarr[0,0].imshow(img)
    axarr[0,0].set_title('Original Image')

    masked_color_array = color_mask(color_data, EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD)
    masked_color_data = np.resize(masked_color_array, color_size_conversion(img.size))
    axarr[0,1].imshow(masked_color_data)
    axarr[0,1].set_title('Masked Image')

    masked_color_vals_only = [val for val in masked_color_array if val != (0,0,0)]
    red, green, blue = [list(color) for color in zip(*masked_color_vals_only)]
    axarr[1,0].hist(blue, COLOR_CHANNEL_MAX)
    axarr[1,0].hold(True)
    axarr[1,0].hist(green, COLOR_CHANNEL_MAX)
    axarr[1,0].hist(red, COLOR_CHANNEL_MAX)
    axarr[1,0].set_xlabel('Color Intensity')
    axarr[1,0].set_ylabel('Number of Pixels')
    axarr[1,0].set_title('Masked Color Histogram')

    masked_img_data = mask(img_data, EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD)
    intensity_vals = flatten([[x for x in row if x != 0] for row in masked_img_data])
    axarr[1,1].hist(intensity_vals, COLOR_CHANNEL_MAX/2)
    axarr[1,1].set_xlabel('Intensity')
    axarr[1,1].set_ylabel('Number of Pixels')
    axarr[1,1].set_title('Masked Image Intensity Histogram')

    plt.show()


def main():
    single_image_analysis('./datasets/IMG_0362.JPG')


main()

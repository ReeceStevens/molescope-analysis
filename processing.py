from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import utility as u
from utility import EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD, COLOR_CHANNEL_MAX


def image_avg_intensity(image_array):
    masked_image = u.mask(image_array, EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD)
    nonzero_count = 0
    for row in masked_image:
        for val in row:
            if val != 0:
                nonzero_count += 1
    return np.sum(masked_image) / nonzero_count


def plot_color_histogram(figure, red, green, blue):
    figure.hist(blue, COLOR_CHANNEL_MAX, alpha=0.5)
    figure.hold(True)
    figure.hist(green, COLOR_CHANNEL_MAX, alpha=0.5)
    figure.hist(red, COLOR_CHANNEL_MAX, alpha=0.5)
    figure.set_xlabel('Color Intensity')
    figure.set_ylabel('Number of Pixels')
    figure.set_title('Masked Color Histogram')


def plot_color_histogram_diff(figure, rgb1, rgb2):
    red1, green1, blue1 = rgb1
    red2, green2, blue2 = rgb2
    u.color_histogram_diff(figure, blue1, blue2)
    figure.hold(True)
    u.color_histogram_diff(figure, green1, green2)
    u.color_histogram_diff(figure, red1, red2)
    figure.set_xlabel('Color Intensity')
    figure.set_ylabel('Difference in Number of Pixels')
    figure.set_title('Color Distribution Differential')


def plot_intensity_histogram(figure, raw_color_data, dimensions):
    bw_data = list(map(u.intensity, list(raw_color_data)))
    img_data = np.resize(bw_data, u.size_inversion(dimensions))
    masked_img_data = u.mask(img_data, EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD)
    intensity_vals = u.flatten([[x for x in row if x != 0] for row in masked_img_data])

    figure.hist(intensity_vals, COLOR_CHANNEL_MAX/2, alpha=0.5)
    figure.set_xlabel('Intensity')
    figure.set_ylabel('Number of Pixels')
    figure.set_title('Masked Image Intensity Histogram')


def plot_original_image(figure, img):
    figure.imshow(img)
    figure.set_title('Original Image')


def plot_masked_color_image(figure, masked_color_data):
    figure.imshow(masked_color_data)
    figure.set_title('Masked Image')


def single_image_analysis(filepath):
    img = Image.open(filepath)
    raw_color_data = img.getdata()

    masked_color_array = u.color_mask(raw_color_data, EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD)
    masked_color_data = np.resize(masked_color_array, u.color_size_conversion(img.size))

    masked_color_vals_only = [val for val in masked_color_array if val != (0,0,0)]
    red, green, blue = [list(color) for color in zip(*masked_color_vals_only)]

    f, axarr = plt.subplots(2,2)
    f.suptitle('Single Image Analysis for ' + filepath)
    plot_original_image(axarr[0,0], img)
    plot_masked_color_image(axarr[0,1], masked_color_data)
    plot_color_histogram(axarr[1,0], red, green, blue)
    plot_intensity_histogram(axarr[1,1], raw_color_data, img.size)
    plt.show()


def two_image_analysis(file1, file2):
    img1 = Image.open(file1)
    img2 = Image.open(file2)

    raw_color_data1 = img1.getdata()
    raw_color_data2 = img2.getdata()
    masked_color_array1 = u.color_mask(raw_color_data1, EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD)
    masked_color_data1 = np.resize(masked_color_array1, u.color_size_conversion(img1.size))
    masked_color_array2 = u.color_mask(raw_color_data2, EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD)
    masked_color_data2 = np.resize(masked_color_array2, u.color_size_conversion(img2.size))

    rgb_array1 = [list(color) for color in zip(*masked_color_array1)]
    rgb_array2 = [list(color) for color in zip(*masked_color_array2)]

    f, axarr = plt.subplots(2,3)
    f.suptitle('Two Image Analysis for ' + file1 + ' and ' + file2)
    plot_original_image(axarr[0,0], img1)
    plot_original_image(axarr[1,0], img2)
    plot_masked_color_image(axarr[0,1], masked_color_data1)
    plot_masked_color_image(axarr[1,1], masked_color_data2)
    plot_color_histogram_diff(axarr[0,2], rgb_array1, rgb_array2)
    plot_intensity_histogram(axarr[1,2], raw_color_data1, img1.size);
    axarr[1,2].hold(True)
    plot_intensity_histogram(axarr[1,2], raw_color_data2, img2.size);

    plt.show()



def main():
    # single_image_analysis('./datasets/IMG_0362.JPG')
    two_image_analysis('./datasets/IMG_0362.JPG', './datasets/IMG_0363.JPG')


main()

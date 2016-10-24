from PIL import Image
import numpy as np
from scipy import stats
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
    figure.hist(blue, COLOR_CHANNEL_MAX, alpha=0.7)
    figure.hold(True)
    figure.hist(green, COLOR_CHANNEL_MAX, alpha=0.7)
    figure.hist(red, COLOR_CHANNEL_MAX, alpha=0.7)
    figure.set_xlabel('Color Intensity')
    figure.set_ylabel('Number of Pixels')
    figure.set_title('Masked Color Histogram')


def plot_color_histogram_diff(figure, rgb1, rgb2):
    red1, green1, blue1 = rgb1
    red2, green2, blue2 = rgb2
    blue_vals, blue_bins = u.color_histogram_diff(figure, blue1, blue2)
    figure.hold(True)
    green_vals, green_bins = u.color_histogram_diff(figure, green1, green2)
    red_vals, red_bins = u.color_histogram_diff(figure, red1, red2)
    figure.set_xlabel('Color Intensity')
    figure.set_ylabel('Difference in Number of Pixels')
    figure.set_title('Color Distribution Differential')
    return [(red_vals, red_bins), (green_vals, green_bins), (blue_vals, blue_bins)]


def get_intensity_histogram(raw_color_data):
    bw_data = list(map(u.intensity, list(raw_color_data)))
    bw_data_mask_removed = [x for x in bw_data if x != 0]
    return np.histogram(bw_data_mask_removed, bins=COLOR_CHANNEL_MAX)


def plot_intensity_histogram(figure, raw_color_data, dimensions):
    bw_data = list(map(u.intensity, list(raw_color_data)))
    img_data = np.resize(bw_data, u.size_inversion(dimensions))
    masked_img_data = u.mask(img_data, EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD)
    intensity_vals = u.flatten([[x for x in row if x != 0] for row in masked_img_data])
    vals, bins = np.histogram(intensity_vals, bins=COLOR_CHANNEL_MAX-EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD, range=(EFFECTIVE_IMAGE_BOUNDARY_THRESHOLD,COLOR_CHANNEL_MAX))
    figure.bar(bins[1:], vals, alpha=0.7)
    figure.set_xlabel('Intensity')
    figure.set_ylabel('Number of Pixels')
    figure.set_title('Masked Image Intensity Histogram')
    return vals, bins



def plot_original_image(figure, img):
    figure.imshow(img)
    figure.set_title('Original Image')


def plot_masked_color_image(figure, masked_color_data):
    figure.imshow(masked_color_data)
    figure.set_title('Masked Image')


def intensity_analysis(figure, hist_data1, hist_data2):
    vals1, bins1 = hist_data1 
    vals2, bins2 = hist_data2

    std1 = np.std(vals1)
    mean1 = np.mean(vals1)
    range1 = np.max(vals1) - np.min(vals1)
    std2 = np.std(vals2)
    mean2 = np.mean(vals2)
    range2 = np.max(vals2) - np.min(vals2)

    ax = figure.axes
    figure.set_title('Intensity Histograms')
    figure.set_xlabel('Intensity Value')
    figure.set_ylabel('Number of Pixels')
    stats1 = 'Image 1:\nMean: ' + str(mean1) + '\nStd Dev: ' + str(std1) + '\nRange: ' + str(range1)
    stats2 = 'Image 2:\nMean: ' + str(mean2) + '\nStd Dev: ' + str(std2) + '\nRange: ' + str(range2)
    tstat, pval = stats.ttest_ind(vals1, vals2)
    figure.text(0.8,0.9,stats1, ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.7))
    figure.text(0.8,0.6,stats2, ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.7))
    figure.text(0.8,0.3,'T-Test statistic: ' + str(tstat) + '\nP-Value: ' + str(pval), ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.7))


def color_analysis(figure, red_data, green_data, blue_data):
    vals1, bins1 = red_data
    vals2, bins2 = green_data
    vals3, bins3 = blue_data

    std1 = np.std(vals1)
    mean1 = np.mean(vals1)
    range1 = np.max(vals1) - np.min(vals1)
    std2 = np.std(vals2)
    mean2 = np.mean(vals2)
    range2 = np.max(vals2) - np.min(vals2)
    std3 = np.std(vals3)
    mean3 = np.mean(vals3)
    range3 = np.max(vals3) - np.min(vals3)

    ax = figure.axes
    figure.set_title('Color Differential Histograms')
    figure.set_xlabel('Intensity Value')
    figure.set_ylabel('Number of Pixels')
    stats1 = 'Red:\nMean: ' + str(mean1) + '\nStd Dev: ' + str(std1) + '\nRange: ' + str(range1)
    stats2 = 'Green:\nMean: ' + str(mean2) + '\nStd Dev: ' + str(std2) + '\nRange: ' + str(range2)
    stats3 = 'Blue:\nMean: ' + str(mean3) + '\nStd Dev: ' + str(std3) + '\nRange: ' + str(range3)
    figure.text(0.8,0.9,stats1, ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.7))
    figure.text(0.8,0.6,stats2, ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='green', alpha=0.7))
    figure.text(0.8,0.3,stats3, ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='blue', alpha=0.7))


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
    red_hist, green_hist, blue_hist = plot_color_histogram_diff(axarr[0,2], rgb_array1, rgb_array2)
    color_analysis(axarr[0,2], red_hist, green_hist, blue_hist)
    vals1, bins1 = plot_intensity_histogram(axarr[1,2], raw_color_data1, img1.size);
    axarr[1,2].hold(True)
    vals2, bins2 = plot_intensity_histogram(axarr[1,2], raw_color_data2, img2.size);
    intensity_analysis(axarr[1,2], (vals1, bins1), (vals2, bins2))

    plt.show()



def main():
    # single_image_analysis('./datasets/IMG_0362.JPG')
    # intensity_analysis('./datasets/IMG_0362.JPG', './datasets/IMG_0363.JPG')
    two_image_analysis('./datasets/presentation_data/iphone3.JPG', './datasets/presentation_data/iphone4.JPG')
    two_image_analysis('./datasets/presentation_data/iphone1.JPG', './datasets/presentation_data/Nexus1.jpg')


main()

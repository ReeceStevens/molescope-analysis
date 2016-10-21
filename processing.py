from functools import partial

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def intensity(rgb_val):
    red, green, blue = rgb_val
    return (red + green + blue) / 3


def size_inversion(init_size):
    length, width = init_size
    return (width, length)


def array_from_jpeg(filepath):
    img = Image.open("./IMG_0345.JPG")
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


def main():
    data = array_from_jpeg('./IMG_0345.JPG')
    masked_data = mask(data, 110)
    plt.imshow(masked_data, 'gist_gray')
    plt.show()

main()

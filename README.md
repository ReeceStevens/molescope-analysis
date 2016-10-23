# Molescope Image Analysis

*BME Senior Design Team 10: Stephanie Teoh, Malvika Gupta, Stefan Hubner,
Reece Stevens*

This repository contains python utilities to analyse images taken with the
molescope device developed by our team.

## Objectives

Our device is designed to maximize image consistency across devices, allowing
images to be medically comparable regardless of the device used. In order to
determine if our design meets these requirements, a quantitative evaluation of
image consistency must be developed.

## Consistency Metrics

1. Size of the sampling area in pixels

2. Image intensity per pixel *within the sampling area*

    - Mean

    - Standard deviation

    - Range

3. Color distribution histogram

## Installation

Installation requires GNU Make and Python 3.5 to be installed on your computer.

Simply type `make install` to install the requisite dependencies in your
computer. Note that you may need administrative privileges if you are trying to
install these packages in your global Python setup. If that makes you feel
not-so-good, activate a virtual environment and a `make install` will have
everything installed locally.

The program can be run by typing `make`. There is some minimal dependency
checking that happens, but be sure to follow the above step first!

## Program Dependencies

Requires Python >= 3.5. The following packages are installed when you type
`make install`:

- `pillow`: Python imaging library

- `numpy`: matrix manipulation and advanced mathematics library

- `matplotlib`: plotting library (MATLAB-like syntax)

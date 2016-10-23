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

## Program Dependencies

Requires Python >= 3.5 and the following packages:

- `pillow`: Python imaging library

- `numpy`: matrix manipulation and advanced mathematics library

- `matplotlib`: plotting library (MATLAB-like syntax)

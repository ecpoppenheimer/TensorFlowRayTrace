"""
Based on 
  http://www.physics.sfasu.edu/astro/color/spectra.html
  RGB VALUES FOR VISIBLE WAVELENGTHS   by Dan Bruton (astro@tamu.edu)
"""

import numpy as np

select = np.select
power = np.power
transpose = np.transpose
arange = np.arange


def factor(wl):
    return select(
        [wl > 700.0, wl < 420.0, True],
        [
            0.3 + 0.7 * (780.0 - wl) / (780.0 - 700.0),
            0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0),
            1.0,
        ],
    )


def raw_r(wl):
    return select(
        [wl >= 580.0, wl >= 510.0, wl >= 440.0, wl >= 380.0, True],
        [1.0, (wl - 510.0) / (580.0 - 510.0), 0.0, (wl - 440.0) / (380.0 - 440.0), 0.0],
    )


def raw_g(wl):
    return select(
        [wl >= 645.0, wl >= 580.0, wl >= 490.0, wl >= 440.0, True],
        [0.0, (wl - 645.0) / (580.0 - 645.0), 1.0, (wl - 440.0) / (490.0 - 440.0), 0.0],
    )


def raw_b(wl):
    return select(
        [wl >= 510.0, wl >= 490.0, wl >= 380.0, True],
        [0.0, (wl - 510.0) / (490.0 - 510.0), 1.0, 0.0],
    )


gamma = 0.80


def correct_r(wl):
    return power(factor(wl) * raw_r(wl), gamma)


def correct_g(wl):
    return power(factor(wl) * raw_g(wl), gamma)


def correct_b(wl):
    return power(factor(wl) * raw_b(wl), gamma)


ww = arange(380.0, 781.0)


def rgb():
    return transpose([correct_r(ww), correct_g(ww), correct_b(ww)])

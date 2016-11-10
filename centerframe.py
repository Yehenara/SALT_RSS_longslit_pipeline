#!/usr/bin/env python

import numpy
import astropy.io.fits as fits
import os
import sys
import math
from optparse import OptionParser
import pysalt.mp_logging
import logging
import scipy.ndimage

import prep_science

def flip(output_coords,
         midline,
        in_center_x, in_center_y,
        sin_rot, cos_rot,
        out_center_x, out_center_y,
         ):

    y_from_midline = output_coords[1] - midline

    # leave x untouched
    in_x = output_coords[0]

    # flip y
    #in_y = output_coords[1]-1 #midline - y_from_midline
    in_y = midline - y_from_midline

    #rel_x = (output_coords[0] - out_center_x)
    #rel_y = output_coords[1]

    #in_x = cos_rot*rel_x - sin_rot*rel_y + in_center_x
    #in_y = sin_rot*rel_x + cos_rot*rel_y + in_center_y
    return (in_x, in_y)



def tilted(p, data):

    setup = {
        'midline': p[0],
        'in_center_x': 0,
        'in_center_y': 0,
        'sin_rot': 0,
        'cos_rot': 0,
        'out_center_x': 0,
        'out_center_y': 0,
    }
    out = scipy.ndimage.interpolation.geometric_transform(
        input=data,
        mapping=flip,
        output_shape=data.shape,
        order=1,
        mode='constant',
        cval=numpy.NaN,
        prefilter=False,
        extra_keywords=setup
        )
    return out

def residuals(p, data):
    print "computing flipped image for center line",p
    flipped = tilted(p, data)
    diff = (data - flipped) / (data+flipped) # weight with image to put more weight on lines
    good_diff = diff[numpy.isfinite(diff)]
    print "diff-size:", good_diff.size
    if (good_diff.size <= 0):
        return [-1.]

    return good_diff


def find_center_line_tilt(data):

    p_init = [1070] #data.shape[1]/2]

    fit = scipy.optimize.leastsq(
        func=residuals,
        x0=p_init,
        args=(data),
        full_output=1,
        epsfcn=1.e-5,
        xtol=1.e-5, # tolerance ~ 0.1 pixel
    )
    p_fit = fit[0]
    print p_fit
    print fit

    return p_fit[0]


if __name__ == "__main__":


    logger = pysalt.mp_logging.setup_logging()

    parser = OptionParser()
    (options, cmdline_args) = parser.parse_args()

    fits_filename = cmdline_args[0]
    hdulist = fits.open(fits_filename)

    data = hdulist['SCI'].data

    sky_lines, sky_continuum = prep_science.filter_isolate_skylines(data)

    chip = sky_lines[:, 0:1024].T
    print chip.shape

    ml = 1076.5

    # fits.PrimaryHDU(data=chip).writeto("flip_raw.fits", clobber=True)
    # fits.PrimaryHDU(data=tilted([ml], chip)).writeto("flip%d.fits" % (ml), clobber=True)
    #
    # center_tilt = find_center_line_tilt(chip)
    #
    # print "best fit:", center_tilt
    # ml = center_tilt
    tilted_sky = tilted([ml], chip)
    fits.PrimaryHDU(data=tilted_sky).writeto("flip%d.fits" % (ml), clobber=True)
    fits.PrimaryHDU(data=(chip-tilted_sky)).writeto("flip%d-diff.fits" % (ml), clobber=True)

    pysalt.mp_logging.shutdown_logging(logger)

#!/usr/bin/env python

import numpy
import astropy.io.fits as fits
import os
import sys

if __name__ == "__main__":

    fits_fn = sys.argv[1]

    y1 = int(sys.argv[2])
    y2 = int(sys.argv[3])

    wl0 = float(sys.argv[4])
    dwl = float(sys.argv[5])
    wlmax = float(sys.argv[6])

    out_fn = sys.argv[7]


    #
    # open fits file
    #
    hdu = fits.open(fits_fn)
    hdu.info()

    # load image data and wavelength map
    img_data_full = hdu['SKYSUB.OPT'].data
    wl_data_full = hdu['WAVELENGTH'].data

    #
    # extract data
    #
    img_data = img_data_full[(y1-1):y2, :]
    wl_data = wl_data_full[(y1-1):y2, :]
    print img_data.shape, wl_data.shape

    # Now prepare the output array
    out_wl_count = int((wlmax - wl0) / dwl) + 1
    print "output: %d wavelength points from %f to %f in steps of %f" % (
        out_wl_count, wl0, wlmax, dwl)
    out_wl = numpy.arange(out_wl_count, dtype=numpy.float) * dwl + wl0
    print out_wl

    #
    # prepare a padded WL array
    #
    wl_data_padded = numpy.pad(wl_data, ((0,0), (1,1)), mode='linear_ramp')
    print wl_data.shape, wl_data_padded.shape

    #img_data_per_wl


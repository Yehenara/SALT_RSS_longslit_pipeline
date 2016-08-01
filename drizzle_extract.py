#!/usr/bin/env python

import numpy
import astropy.io.fits as fits
import os
import sys
import math

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
    # prepare the output flux array, and initialize with NaNs
    out_flux = numpy.zeros_like(out_wl)
    out_flux[:] = numpy.NaN
    print out_wl

    #
    # prepare a padded WL array
    #
    wl_data_padded = numpy.pad(wl_data, ((0,0), (1,1)), mode='edge') #linear_ramp')
    print wl_data.shape, wl_data_padded.shape
    wl_width = 0.5*(wl_data_padded[:, 2:] - wl_data_padded[:, :-2])
    wl_from = 0.5*(wl_data_padded[:, 0:-2] + wl_data_padded[:,1:-1])
    wl_to = 0.5*(wl_data_padded[:, 1:-1] + wl_data_padded[:,2:])
    print wl_width.shape, wl_from.shape, wl_to.shape

    # compute the flux density array (i.e. flux per wavelength) by dividing the
    # flux image (which is flux integrated over the width of a pixel) by the width of each pixel
    img_data_per_wl = img_data / wl_width

    #
    # now drizzle the data into the output container
    #
    wl_width_1d = wl_width.ravel()
    wl_from_1d = wl_from.ravel()
    wl_to_1d = wl_to.ravel()
    wl_data_1d = wl_data.ravel()
    img_data_per_wl_1d = img_data_per_wl.ravel()

    for px in range(wl_width_1d.shape[0]):

        # find first and last pixel in output array to receive some of the flux of this input pixel
        first_px = (wl_from_1d[px] - wl0) / dwl
        last_px = (wl_to_1d[px] - wl0) / dwl

        pixels = range(int(math.floor(first_px)), int(math.ceil(last_px))) #, dtype=numpy.int)
        #print wl_data_1d[px], wl_width_1d[px], first_px, last_px, pixels
        for i, tp in enumerate(pixels):

            if (i == 0 and i == len(pixels)-1):
                # all flux in a single output pixel
                fraction = last_px - first_px
            elif (i == 0):
                # first pixel, with more to come
                fraction = (tp + 1.) - first_px
            elif (i == len(pixels)-1):
                # last of many pixels
                fraction = last_px - tp
            else:
                # some pixel in the middle
                fraction = 1.

            if (numpy.isnan(out_flux[tp])):
                out_flux[tp] = 0.
            out_flux[tp] += fraction * img_data_per_wl_1d[px] * dwl
            #print "     ", tp, fraction


    #
    # Finally, merge wavelength data and flux and write output to file
    #
    numpy.savetxt(out_fn,
                  numpy.append(out_wl.reshape((-1,1)),
                               out_flux.reshape((-1,1)),
                               axis=1))
    numpy.savetxt(out_fn+".perwl",
                  numpy.append(wl_data_1d.reshape((-1,1)),
                               img_data_per_wl_1d.reshape((-1,1)), axis=1))
    numpy.savetxt(out_fn + ".raw",
              numpy.append(wl_data_1d.reshape((-1, 1)),
                           img_data.reshape((-1, 1)), axis=1))

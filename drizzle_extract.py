#!/usr/bin/env python

import numpy
import astropy.io.fits as fits
import os
import sys
import math
from optparse import OptionParser
import pysalt.mp_logging

def drizzle_extract_spectrum(fits_fn, out_fn,
                             input_ext="SKYSUB.OPT",
                             minwl=-1, maxwl=-1, dwl=-1,
                             y_ranges=None,
                             ):

    # y1 = int(sys.argv[2])
    # y2 = int(sys.argv[3])
    #
    # wl0 = float(sys.argv[4])
    # dwl = float(sys.argv[5])
    # wlmax = float(sys.argv[6])
    #
    # out_fn = sys.argv[7]


    #
    # open fits file
    #
    hdu = fits.open(fits_fn)
    hdu.info()

    # load image data and wavelength map
    img_data_full = hdu[input_ext].data
    wl_data_full = hdu['WAVELENGTH'].data

    wl0 = minwl
    wlmax = maxwl
    if (minwl < 0):
        wl0 = numpy.min(wl_data_full)
    if (maxwl < 0):
        wlmax = numpy.max(wl_data_full)
    if (dwl < 0):
        _disp = numpy.diff(wl_data_full)
        dwl = 0.5*numpy.min(_disp)
        if (dwl <= 0):
            dwl = 0.1

    # Now prepare the output array
    out_wl_count = int((wlmax - wl0) / dwl) + 1
    print "output: %d wavelength points from %f to %f in steps of %f" % (
        out_wl_count, wl0, wlmax, dwl)
    out_wl = numpy.arange(out_wl_count, dtype=numpy.float) * dwl + wl0

    spectra_1d = numpy.empty((out_wl_count, len(y_ranges)))

    for cur_yrange, _yrange in enumerate(y_ranges):
        y1 = _yrange[0]
        y2 = _yrange[1]

        #
        # extract data
        #
        img_data = img_data_full[(y1-1):y2, :]
        wl_data = wl_data_full[(y1-1):y2, :]
        print img_data.shape, wl_data.shape

        #
        # prepare the output flux array, and initialize with NaNs
        #
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
        # done with this 1-d extracted spectrum
        #
        spectra_1d[:,cur_yrange] = out_flux[:]



    #
    # Finally, merge wavelength data and flux and write output to file
    #
    numpy.savetxt(out_fn,
                  numpy.append(out_wl.reshape((-1,1)),
                               spectra_1d, axis=1))
    # numpy.savetxt(out_fn+".perwl",
    #               numpy.append(wl_data_1d.reshape((-1,1)),
    #                            img_data_per_wl_1d.reshape((-1,1)), axis=1))
    # numpy.savetxt(out_fn + ".raw",
    #           numpy.append(wl_data_1d.reshape((-1, 1)),
    #                        img_data.reshape((-1, 1)), axis=1))



if __name__ == "__main__":


    logger = pysalt.mp_logging.setup_logging()

    parser = OptionParser()
    parser.add_option("", "--input", dest="inputext",
                      help="name of extension to extract",
                      default="SKYSUB.OPT")
    parser.add_option("-n", "--minwl", dest="minwl",
                      help="minimum wavelength of output",
                      default=-1, type=float)
    parser.add_option("-x", "--maxwl", dest="maxwl",
                      help="maximum wavelength of output",
                      default=-1, type=float)
    parser.add_option("-d", "--dwl", dest="dwl",
                      help="wavelength sampling / dispersion",
                      default=-1, type=float)
    # parser.add_option("-s", "--scale", dest="skyscaling",
    #                   help="How to scale the sky spectrum (none,s2d,p2d)",
    #                   default="none")
    # parser.add_option("-d", "--debug", dest="debug",
    #                    action="store_true", default=False)
    (options, cmdline_args) = parser.parse_args()

    print options
    print cmdline_args

    fits_fn = cmdline_args[0]
    out_fn = cmdline_args[1]

    y_ranges = []
    for inp in cmdline_args[2:]:
        items = inp.split(":")
        y_ranges.append([int(items[0]), int(items[1])])

    drizzle_extract_spectrum(fits_fn=fits_fn,
                             out_fn=out_fn,
                             input_ext=options.inputext,
                             minwl=options.minwl,
                             maxwl=options.maxwl,
                             dwl=options.dwl,
                             y_ranges=y_ranges,
                             )

    pysalt.mp_logging.shutdown_logging(logger)



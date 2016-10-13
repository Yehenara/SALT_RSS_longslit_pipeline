#!/usr/bin/env python

import numpy
import astropy.io.fits as fits
import os
import sys
import math
from optparse import OptionParser
import pysalt.mp_logging
import logging



def drizzle_spec(img_data, wl, var_data, out_wl):

    if (img_data.ndim == 1):
        img_data = img_data.reshape((-1,1))
    if (wl.ndim == 1):
        wl = wl.reshape((-1,1))
    if (var_data.ndim == 1):
        var_data = var_data.reshape((-1,1))

    #
    # prepare the output flux array, and initialize with NaNs
    #
    out_flux = numpy.zeros_like(out_wl)
    out_flux[:] = numpy.NaN
    out_var = numpy.zeros_like(out_wl)
    out_var[:] = numpy.NaN
    #print out_wl

    #
    # prepare a padded WL array
    #
    wl_data_padded = numpy.pad(wl, ((0, 0), (1, 1)), mode='edge') #linear_ramp')
    #print wl_data.shape, wl_data_padded.shape
    wl_width = 0.5*(wl_data_padded[:, 2:] - wl_data_padded[:, :-2])
    wl_from = 0.5*(wl_data_padded[:, 0:-2] + wl_data_padded[:,1:-1])
    wl_to = 0.5*(wl_data_padded[:, 1:-1] + wl_data_padded[:,2:])
    #print wl_width.shape, wl_from.shape, wl_to.shape

    # compute the flux density array (i.e. flux per wavelength) by dividing the
    # flux image (which is flux integrated over the width of a pixel) by the width of each pixel
    img_data_per_wl = img_data / wl_width
    var_data_per_wl = var_data / wl_width


    #
    # now drizzle the data into the output container
    #
    wl_width_1d = wl_width.ravel()
    wl_from_1d = wl_from.ravel()
    wl_to_1d = wl_to.ravel()
    wl_data_1d = wl.ravel()
    img_data_per_wl_1d = img_data_per_wl.ravel()
    var_data_1d = var_data.ravel()
    var_data_per_wl_1d = var_data_per_wl.ravel()
    wl_width_1d = wl_width.ravel()

    wl0 = out_wl[0]
    dwl = out_wl[1] - out_wl[0]

    for px in range(wl_width_1d.shape[0]):

        # find first and last pixel in output array to receive some of the flux of this input pixel
        first_px = (wl_from_1d[px] - wl0) / dwl
        last_px = (wl_to_1d[px] - wl0) / dwl

        pixels = range(int(math.floor(first_px)), int(math.ceil(last_px)))  # , dtype=numpy.int)
        # print wl_data_1d[px], wl_width_1d[px], first_px, last_px, pixels
        for i, tp in enumerate(pixels):

            if (i == 0 and i == len(pixels) - 1):
                # all flux in a single output pixel
                fraction = last_px - first_px
            elif (i == 0):
                # first pixel, with more to come
                fraction = (tp + 1.) - first_px
            elif (i == len(pixels) - 1):
                # last of many pixels
                fraction = last_px - tp
            else:
                # some pixel in the middle
                fraction = 1.

            if (numpy.isnan(out_flux[tp])):
                out_flux[tp] = 0.
                out_var[tp] = 0.
            out_flux[tp] += fraction * img_data_per_wl_1d[px]  # * (dwl / wl_width_1d[px])
            out_var[tp] += fraction * var_data_per_wl_1d[px]  # * (dwl / wl_width_1d[px])
            # print "     ", tp, fraction

    return out_flux, out_var





def drizzle_extract_spectrum(fits_fn, out_fn,
                             input_ext="SKYSUB.OPT",
                             minwl=-1, maxwl=-1, dwl=-1,
                             y_ranges=None,
                             output_format="fits",
                             ):

    logger = logging.getLogger("DrizzleSpec")

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
    logger.info("opening input fits %s" % (fits_fn))
    #hdu.info()

    # load image data and wavelength map
    img_data_full = hdu[input_ext].data
    wl_data_full = hdu['WAVELENGTH'].data
    variance_full = hdu['VAR'].data

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
    logger.info("output: %d wavelength points from %f to %f in steps of %f angstroems" % (
        out_wl_count, wl0, wlmax, dwl))
    out_wl = numpy.arange(out_wl_count, dtype=numpy.float) * dwl + wl0

    spectra_1d = numpy.empty((out_wl_count, len(y_ranges)))
    variance_1d = numpy.empty((out_wl_count, len(y_ranges)))

    for cur_yrange, _yrange in enumerate(y_ranges):
        y1 = _yrange[0]
        y2 = _yrange[1]
        logger.info("Extracting 1-D spectrum from columns %d --- %d" % (y1,y2))
        #
        # extract data
        #
        img_data = img_data_full[(y1-1):y2, :]
        wl_data = wl_data_full[(y1-1):y2, :]
        var_data = variance_full[(y1-1):y2, :]
        #print img_data.shape, wl_data.shape

        #
        # prepare the output flux array, and initialize with NaNs
        #
        out_flux = numpy.zeros_like(out_wl)
        out_flux[:] = numpy.NaN
        out_var = numpy.zeros_like(out_wl)
        out_var[:] = numpy.NaN
        #print out_wl

        #
        # prepare a padded WL array
        #
        wl_data_padded = numpy.pad(wl_data, ((0,0), (1,1)), mode='edge') #linear_ramp')
        #print wl_data.shape, wl_data_padded.shape
        wl_width = 0.5*(wl_data_padded[:, 2:] - wl_data_padded[:, :-2])
        wl_from = 0.5*(wl_data_padded[:, 0:-2] + wl_data_padded[:,1:-1])
        wl_to = 0.5*(wl_data_padded[:, 1:-1] + wl_data_padded[:,2:])
        #print wl_width.shape, wl_from.shape, wl_to.shape

        # compute the flux density array (i.e. flux per wavelength) by dividing the
        # flux image (which is flux integrated over the width of a pixel) by the width of each pixel
        img_data_per_wl = img_data / wl_width
        var_data_per_wl = var_data / wl_width

        #
        # now drizzle the data into the output container
        #
        wl_width_1d = wl_width.ravel()
        wl_from_1d = wl_from.ravel()
        wl_to_1d = wl_to.ravel()
        wl_data_1d = wl_data.ravel()
        img_data_per_wl_1d = img_data_per_wl.ravel()
        var_data_1d = var_data.ravel()
        var_data_per_wl_1d = var_data_per_wl.ravel()
        wl_width_1d = wl_width.ravel()

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
                    out_var[tp] = 0.
                out_flux[tp] += fraction * img_data_per_wl_1d[px] #* (dwl / wl_width_1d[px])
                out_var[tp] += fraction * var_data_per_wl_1d[px] #* (dwl / wl_width_1d[px])
                #print "     ", tp, fraction


        #
        # done with this 1-d extracted spectrum
        #
        spectra_1d[:,cur_yrange] = out_flux[:]
        variance_1d[:, cur_yrange] = out_var[:]
        logger.debug("Done with 1-d extraction")

    #
    # Finally, merge wavelength data and flux and write output to file
    #
    if (output_format.lower() == "fits"):
        logger.info("Writing FITS output to %s" % (out_fn))
        # create the output multi-extension FITS
        spec_3d = numpy.empty((2, spectra_1d.shape[1],  spectra_1d.shape[0]))
        spec_3d[0,:,:] = spectra_1d.T[:,:]
        spec_3d[1,:,:] = variance_1d.T[:,:]

            # numpy.append(spectra_1d.reshape((spectra_1d.shape[1], spectra_1d.shape[0], 1)),
            #                    variance_1d.reshape((spectra_1d.shape[1], spectra_1d.shape[0], 1)),
            #                     axis=2,
            #                    )
        print spec_3d.shape
        hdulist = fits.HDUList([
            fits.PrimaryHDU(header=hdu[0].header),
            fits.ImageHDU(data=spectra_1d.T, name="SCI"),
            fits.ImageHDU(data=variance_1d.T, name="VAR"),
            fits.ImageHDU(data=spec_3d, name="SPEC3D")
        ])
        # add headers for the wavelength solution
        for ext in ['SCI', 'VAR']:
            hdulist[ext].header['WCSNAME'] = "calibrated wavelength"
            hdulist[ext].header['CRPIX1'] = 1.
            hdulist[ext].header['CRVAL1'] = wl0
            hdulist[ext].header['CD1_1'] = dwl
            hdulist[ext].header['CTYPE1'] = "AWAV"
            hdulist[ext].header['CUNIT1'] = "Angstrom"
            for i,yr in enumerate(y_ranges):
                keyname = "YR_%03d" % (i+1)
                value = "%04d:%04d" % (yr[0], yr[1])
                hdulist[ext].header[keyname] = (value, "y-range for aperture %d" % (i+1))
        hdulist.writeto(out_fn, clobber=True)
    else:
        logger.info("Writing output as ASCII to %s / %s.var" % (out_fn, out_fn))
        numpy.savetxt(out_fn,
                      numpy.append(out_wl.reshape((-1,1)),
                                   spectra_1d, axis=1))
        numpy.savetxt(out_fn+".var",
                      numpy.append(out_wl.reshape((-1,1)),
                                   variance_1d, axis=1))

    logger.debug("All done!")
    # numpy.savetxt(out_fn+".perwl",
    #               numpy.append(wl_data_1d.reshape((-1,1)),
    #                            img_data_per_wl_1d.reshape((-1,1)), axis=1))
    # numpy.savetxt(out_fn + ".raw",
    #           numpy.append(wl_data_1d.reshape((-1, 1)),
    #                        img_data.reshape((-1, 1)), axis=1))



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", dest="inputext",
                        default="SKYSUB.OPT",
                        type=str,
                        help="name of extension to extract")
    parser.add_argument("-n", "--minwl", dest="minwl",
                        help="minimum wavelength of output",
                        type=float, default=-1)
    parser.add_argument("-x", "--maxwl", dest="maxwl",
                        help="maximum wavelength of output",
                        default=-1, type=float)
    parser.add_argument("-d", "--dwl", dest="dwl",
                        help="wavelength sampling / dispersion",
                        default=-1, type=float)

    parser.add_argument("input_file", type=str,
                        help="input file (reduced by rk_specred)")
    parser.add_argument("output_file", type=str,
                        help="output file (written as both ascii and fits)")
    parser.add_argument("blocks",
                        nargs='+',
                        help="y-range intervals in the format y1:y2")

    args = parser.parse_args()

    logger = pysalt.mp_logging.setup_logging()

    fits_fn = args.input_file
    out_fn = args.output_file

    y_ranges = []
    for inp in args.blocks:
        items = inp.split(":")
        y_ranges.append([int(items[0]), int(items[1])])

    drizzle_extract_spectrum(fits_fn=fits_fn,
                             out_fn=out_fn,
                             input_ext=args.inputext,
                             minwl=args.minwl,
                             maxwl=args.maxwl,
                             dwl=args.dwl,
                             y_ranges=y_ranges,
                             output_format=options.format,
                             )

    pysalt.mp_logging.shutdown_logging(logger)



#!/usr/bin/env python

import astropy.io.fits as fits
import os
import sys
import numpy
from optparse import OptionParser
import logging

import drizzle_extract



if __name__ == "__main__":

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
    parser.add_option("-f", "--format", dest="format",
                      help="output format (FITS/ascii)",
                      default="fits", type=str)
    (options, cmdline_args) = parser.parse_args()

    input_file = cmdline_args[0]
    output_file = cmdline_args[1]

    if (not os.path.isfile(input_file)):
        print "Input file does not exist"
        sys.exit(0)


    hdulist = fits.open(input_file)

    img_data = hdulist['SKYSUB.OPT'].data
    wl_data = hdulist['WAVELENGTH'].data
    var_data = hdulist['VAR'].data

    wl0 = options.minwl
    wlmax = options.maxwl
    dwl = options.dwl
    minwl = options.minwl

    if (minwl < 0):
        wl0 = numpy.min(wl_data)
    if (wlmax < 0):
        wlmax = numpy.max(wl_data)
    if (dwl < 0):
        _disp = numpy.diff(wl_data)
        dwl = numpy.fabs(dwl)*numpy.min(_disp)
        if (dwl <= 0):
            dwl = 0.1

    # Now prepare the output array
    logger = logging.getLogger("RectifySpec")
    out_wl_count = int((wlmax - wl0) / dwl) + 1
    logger.info("output: %d wavelength points from %f to %f in steps of %f angstroems" % (
        out_wl_count, wl0, wlmax, dwl))
    out_wl = numpy.arange(out_wl_count, dtype=numpy.float) * dwl + wl0

    rect_img = numpy.empty((img_data.shape[0], out_wl.shape[0]))
    rect_var = numpy.empty((img_data.shape[0], out_wl.shape[0]))
    rect_img[:,:] = numpy.NaN
    rect_var[:,:] = numpy.NaN

    for y in range(0, img_data.shape[0]):

        sys.stdout.write("\rRectifying line %4d of %4d" % (y+1, img_data.shape[0])); sys.stdout.flush()

        line_img = img_data[y:y+1, :]
        line_wl = wl_data[y:y+1, :]
        line_var = var_data[y:y+1, :]

        drizzled = drizzle_extract.drizzle_spec(line_img, line_wl, line_var, out_wl)
        _data, _var = drizzled

        rect_img[y, :] = _data[:]
        rect_var[y, :] = _var[:]

    print ("\nAll done, preparing output")
    hdus = [None]*3
    for i in range(1,3):
        imghdu = fits.ImageHDU()

        imghdu.header['WCSNAME'] = "wavelength"

        imghdu.header['CRPIX1'] = 1.0
        imghdu.header['CRPIX2'] = 1.0

        imghdu.header['CRVAL1'] = wl0
        imghdu.header['CTYPE1'] = 'AWAV'
        imghdu.header['CUNIT1'] = "Angstrom"
        imghdu.header['CD1_1'] = dwl

        imghdu.header['CRVAL2'] = 1.
        imghdu.header['CTYPE2'] = 'POS'
        imghdu.header['CUNIT2'] = 'pixel'
        imghdu.header['CD2_2'] = 1.0

        hdus[i] = imghdu

    hdus[1].data = rect_img
    hdus[1].name = "SCI.RECT"

    hdus[2].data = rect_var
    hdus[2].name = 'VAR.RECT'

    hdus[0] = fits.PrimaryHDU()

    hdulist = fits.HDUList(hdus)
    hdulist.writeto(output_file, clobber=True)
    #fits.PrimaryHDU(data=rect_img).writeto("rect_img.fits", clobber=True)
    #fits.PrimaryHDU(data=rect_var).writeto("rect_var.fits", clobber=True)
#!/usr/bin/env python


import os
import sys
from astropy.io import fits
import numpy


if __name__ == "__main__":

    infile = sys.argv[1]
    wl = float(sys.argv[2])
    width = int(sys.argv[3])

    hdulist = fits.open(infile)

    wlmap = hdulist['WAVELENGTH'].data

    y,x = numpy.indices(wlmap.shape)
    print y
    print x

    x[wlmap > wl] = 0.
    x_start = numpy.max(x, axis=1)
    print x_start

    # wl_pick = wlmap[:, x_start]
    # print wl_pick
    #
    img_data = hdulist['SCI'].data
    img_data /= hdulist['SKY.SCALE'].data

    img_padded = numpy.pad(img_data,
                           pad_width=((0,0), (width,width)),
                           mode='constant',)
    print img_padded.shape, hdulist['SCI'].data.shape


    x_start += width
    cutout = numpy.empty((wlmap.shape[0], 2*width+1))
    for iy in range(cutout.shape[0]):
        print iy, wlmap[iy,x_start[iy]]
        cutout[iy,:] = img_padded[iy, x_start[iy]-width:x_start[iy]+width+1]
    #cutout = hdulist['SCI'].data[x_start-width:x_start+width]
    print cutout.shape

    fits.PrimaryHDU(data=cutout).writeto("cutout.fits", clobber=True)
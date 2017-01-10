#!/bin/env python


import os
import sys
import numpy
import pyfits
from pysalt import mp_logging
import logging
import scipy
import scipy.interpolate

import traceline

import map_distortions

if __name__ == "__main__":

    logger_setup = mp_logging.setup_logging()
    logger = logging.getLogger("ModelDistortions")


    fn = sys.argv[1]
    hdulist = pyfits.open(fn)

    img_size = hdulist['SCI'].header['NAXIS1']

    skytable_ext = hdulist['SKYLINES']

    n_lines = skytable_ext.header['NAXIS2']
    n_cols = skytable_ext.header['TFIELDS']
    skyline_list = numpy.empty((n_lines, n_cols))
    for i in range(n_cols):
        skyline_list[:,i] = skytable_ext.data.field(i)

    print "        X      peak continuum   c.noise       S/N      WL/X"
    print "="*59
    numpy.savetxt(sys.stdout, skyline_list, "%9.3f")
    print "=" * 59

    good_lines = numpy.isfinite(skyline_list[:,0]) & (skyline_list[:,4]>5)
    # good_lines = traceline.pick_line_every_separation(
    #     skyline_list,
    #     trace_every=5,
    #     min_line_separation=40,
    #     n_pixels=img_size,
    #     min_signal_to_noise=10,
    #     )
    print "Selecting %d of %d lines to compute distortion model" % (good_lines.shape[0], n_lines)
    skyline_list = skyline_list[good_lines]


    print "\n"*5
    print "        X      peak continuum   c.noise       S/N      WL/X"
    print "="*59
    numpy.savetxt(sys.stdout, skyline_list, "%9.3f")
    print "=" * 59

    #
    # Now load all files for these lines
    #
    all_lines = [None] * skyline_list.shape[0]
    wl_2d = hdulist['WAVELENGTH'].data
    diff_2d = hdulist['SKYSUB.OPT'].data
    img_2d = hdulist['SCI'].data

    try:
        badrows = hdulist['BADROWS'].data
        badrows = badrows > 0
    except:
        badrows = None

    dist, dist_binned, bias_level, dist_median, dist_std,  = map_distortions.map_distortions(wl_2d=wl_2d,
                                                        diff_2d=diff_2d,
                                                        img_2d=img_2d,
                                                        x_list=skyline_list[:,0],
                                                        y=610, #img_2d.shape[0] / 2.
                                                        badrows=badrows,
                                                        )

    print "\n----------"*10,"mapping distortions", "\n-------------"*5

    readnoise = 7
    gain=2

    regfile = open("distortions.reg", "w")
    print >>regfile, """\
# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
physical
    """

    for i, line in enumerate(skyline_list):

        print >>regfile, "point(%.2f,%.2f) # point=circle" % (line[0], 610)

        #fn = "distortion_%d.bin" % (line[0])
        #linedata = numpy.loadtxt(fn)

        linedata_mean = dist_binned[i]
        linedata_med = dist_median[i]
        linedata_std = dist_std[i]

        # compute median s/n
        s2n = (linedata_med[:, 10] - linedata_med[:, 13]) / (numpy.sqrt((linedata_med[:, 13] * gain) + readnoise ** 2) / gain)
        median_s2n = numpy.median(s2n[s2n>0])
        logger.info("Line @ %f --> median s/n = %f" % (line[0], median_s2n))

        # med_flux = numpy.median(linedata_mean[:, 11][numpy.isfinite(linedata_mean[:, 11])])
        # noise = numpy.sqrt(med_flux**2*gain + readnoise**2)/gain
        if (median_s2n < 4):
            logger.info("Ignoring line at x=%f because of insuffient s/n" % (line[0]))
            continue

        #
        # Also consider the typical scatter in center positions
        #
        median_pos_scatter = numpy.median(linedata_std[:,8])
        logger.info("Median position scatter: %f" % (median_pos_scatter))

        # compute the wavelength error from the actual line position
        linedata_mean[:, 9] -= linedata_mean[:, 7]

        # correct the line for a global error in wavelength
        not_nan = numpy.isfinite(linedata_mean[:, 9])
        med_dwl = numpy.median(linedata_mean[:, 9][not_nan])
        linedata_mean[:, 9] -= med_dwl

        all_lines[i] = linedata_mean

    all_lines = numpy.array(all_lines)
    print all_lines.shape

    wl_dist = all_lines[:, :, [7, 0, 9]] # wl,y,d_wl
    print wl_dist.shape
    x = wl_dist.reshape((-1,wl_dist.shape[2]))
    print x.shape
    numpy.savetxt("distortion_model.in", x)


    #
    # Now convert all the data we have into a full 2-d model in wl & x
    #
    logger.info("Computing 2-D interpolator")
    wl_dist = x[numpy.isfinite(x[:,2])]
    print wl_dist.shape
    interpol = scipy.interpolate.SmoothBivariateSpline(
        x=wl_dist[:,0],
        y=wl_dist[:,1],
        z=wl_dist[:,2],
        kx=1, ky=1,
    )

    numpy.savetxt("interpol_in.x", wl_dist[:,0])
    numpy.savetxt("interpol_in.y", wl_dist[:,1])
    numpy.savetxt("interpol_in.z", wl_dist[:,2])

    #
    # Now compute a full 2-d grid of distortions as fct. of y and wavelength positions
    #
    print "computing 2-d distortion model"
    wlmap =hdulist['WAVELENGTH'].data
    _y,_x = numpy.indices(wlmap.shape)
    distortion_2d = interpol(
        x=wlmap,
        y=_y,
        grid=False,)
    print distortion_2d.shape

    pyfits.PrimaryHDU(data=distortion_2d).writeto("distortion_2d.fits", clobber=True)



    mp_logging.shutdown_logging(logger_setup)

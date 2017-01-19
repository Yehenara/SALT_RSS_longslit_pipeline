#!/bin/env python


import os
import sys
import numpy
import pyfits as fits

import logging
import scipy
import scipy.interpolate

import find_sources
import tracespec

import pysalt.mp_logging


def find_background_correction(hdulist, badrows=None,
                               sources=None, source_ext="SKYSUB.OPT",
                               dimension=1, fitorder=9):

    if (sources is None):
        # compute a source list (includes source position, extent, and intensity)
        prof, prof_var = find_sources.continuum_slit_profile(hdulist)
        numpy.savetxt("source_profile", prof)
        sources = find_sources.identify_sources(prof, prof_var)

        print "="*50,"\n List of sources: \n","="*50
        print sources
        print "\n-----"*5

    if (badrows is None):
        try:
            badrows = (hdulist['BADROWS'].data > 0)
        except KeyError:
            pass

    print "bad-rows:", badrows


    img_data = hdulist[source_ext].data.copy()
    if (badrows is not None):
        img_data[badrows] = numpy.NaN

    #
    # Mask out all sources, being generous to also mask out wings etc
    # out to twice the source extent
    #
    for src in sources:
        w = src[2:4] - src[0]
        img_data[src[0]-2*(src[0]-src[2]):src[0]+2*(src[3]-src[0])] = numpy.NaN

    fits.PrimaryHDU(data=img_data).writeto("zeroback.fits", clobber=True)

    bg_level = numpy.nanmedian(img_data)
    print bg_level

    bg_slit = numpy.nanmedian(img_data, axis=1)
    print bg_slit.shape
    numpy.savetxt("zeroslit", bg_slit)

    iy = numpy.arange(bg_slit.shape[0])

    good_data = numpy.isfinite(bg_slit)
    spline_in_y = iy[good_data]
    spline_in_bg = bg_slit[good_data]
    numpy.savetxt("zerobg.spline.in", numpy.append(spline_in_y.reshape((-1,1)),
                                                   spline_in_bg.reshape((-1,1)), axis=1))


    valid = numpy.isfinite(spline_in_bg)
    polyorder = 9
    for iter in range(3):

        try:
            poly = numpy.polyfit(
                x=spline_in_y[valid],
                y=spline_in_bg[valid],
                deg=polyorder,
            )
        except Exception as e:
            print e


        fit = numpy.polyval(poly, spline_in_y)

        residuals = spline_in_bg - fit

        median = numpy.median(residuals[valid])
        rms = numpy.std(residuals[valid])
        print median, rms

        outlier = (residuals > median+3*rms) | (residuals < median-3*rms)
        valid[outlier] = False

    fit = numpy.polyval(poly, iy)
    #combined = numpy.empty((bg_slit.shape[0], 3))
    combined = numpy.array(
        [iy, bg_slit,fit]
    ).T
    print combined.shape
    numpy.savetxt("zerobg.fit", combined)

    #
    # Compute a full 2-d model for the frame background
    #
    _y, _x = numpy.indices(img_data.shape)
    fullframe_bg = numpy.polyval(poly, _y)

    return fullframe_bg


if __name__ == "__main__":
    log_setup = pysalt.mp_logging.setup_logging()

    fn = sys.argv[1]

    hdulist = fits.open(fn)

    fullframe_bg = find_background_correction(
        hdulist=hdulist,
        badrows=None, sources=None,
    )
    skysub_opt = hdulist['SKYSUB.OPT'].data
    skysub_opt -= fullframe_bg

    fits.PrimaryHDU(data=skysub_opt).writeto(fn[:-5]+".bgsub.fits", clobber=True)

    pysalt.mp_logging.shutdown_logging(log_setup)

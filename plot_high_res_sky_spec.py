#!/usr/bin/env python

import warnings
import StringIO

import os, sys, scipy, scipy.stats, scipy.ndimage, numpy
import scipy.interpolate
import wlcal
import skyline_intensity
import traceline

from astropy.io import fits

#from optimal_spline_basepoints import satisfy_schoenberg_whitney
import optimal_spline_basepoints
import bottleneck
import logging

warnings.simplefilter('ignore', UserWarning)

from wlcal import lineinfo_colidx

import pysalt.mp_logging

import matplotlib.pyplot as pyplot



if __name__ == "__main__":

    fn = sys.argv[1]

    #allskies = numpy.loadtxt(fn)
    #print allskies.shape


    hdulist = fits.open(fn)
    flux = hdulist['SCI'].data
    wl = hdulist['WAVELENGTH'].data

    wl_bad = None
    flux_bad = None

    try:
        good_sky_data = hdulist['GOOD_SKY_DATA'].data.astype(numpy.bool)
        _flux = flux.copy()
        _wl = wl.copy()
        flux = flux[good_sky_data]
        wl = wl[good_sky_data]

        wl_bad = _wl[~good_sky_data]
        flux_bad = _flux[~good_sky_data]

        valid = numpy.isfinite(wl_bad) & numpy.isfinite(flux_bad)
        wl_bad = wl_bad[valid]
        flux_bad = flux_bad[valid]
        print("Using only GOOD_SKY_DATA for plotting")
    except:
        try:
            goodrows = hdulist['BADROWS'].data > 0
            flux = flux[~goodrows]
            wl = wl[~goodrows]
            print("Using only data from rows not in BADROWS")
        except:
            print("Using all pixels")
            pass
            # badrows = numpy.ones_like(flux, dtype=numpy.bool)

    good_data = numpy.isfinite(flux)
    flux = flux[good_data]
    wl = wl[good_data]

    spline = None

    fig = pyplot.figure()
    n_rows = 9

    wl_min = numpy.min(wl)
    wl_max = numpy.max(wl)
    print wl_min, wl_max
    overlap = 0.02 * (wl_max-wl_min) / n_rows

    every = 25
    everyfit = 25**2

    si = numpy.argsort(wl)
    spline = scipy.interpolate.UnivariateSpline(
        x=wl[si].flatten()[::everyfit], y=flux[si].flatten()[::everyfit],
    )

    for row in range(n_rows):
        ax = fig.add_subplot(n_rows, 1, row+1)

        this_wl_min = float(row)/n_rows * (wl_max-wl_min) + wl_min - overlap
        this_wl_max = float(row+1)/n_rows * (wl_max-wl_min) + wl_min + overlap
        print row, "==>", this_wl_min, this_wl_max

        if (wl_bad is not None and flux_bad is not None):
            in_wl_range = (wl_bad >= this_wl_min) & (wl_bad <= this_wl_max)
            this_wl = wl_bad[in_wl_range][::every]
            this_flux = flux_bad[in_wl_range][::every]
            ax.plot(this_wl, this_flux, marker=",", color="#a0a0a0",
                    linestyle='None')


        in_wl_range = (wl >= this_wl_min) & (wl <= this_wl_max)
        this_wl = wl[in_wl_range][::every]
        this_flux = flux[in_wl_range][::every]
        print row, this_wl.shape


        ax.plot(this_wl, this_flux, "b,")
        ax.set_xlim((this_wl_min, this_wl_max))
        ax.set_ylim((0,500))

        print spline
        if (spline is not None):
            highres_wl = numpy.linspace(this_wl_min, this_wl_max, 1000)
            #print highres_wl
            highres_flux = spline(highres_wl)
            ax.plot(highres_wl, highres_flux, 'r-')
            numpy.savetxt("specblock.%d" % (row+1),
                          numpy.array([this_wl, this_flux]).T)
            numpy.savetxt("specblock_fit.%d" % (row+1),
                          numpy.array([highres_wl, highres_flux]).T)
            max_spline_flux = numpy.max(highres_flux)
            ax.set_ylim((0, 1.1*max_spline_flux))

        for label in ax.get_xticklabels():
            label.set_fontsize(5)
            x,y = label.get_position()
            # print x,y
            label.set_position((x,y+0.07))
            #label.set_verticalalignment('top')
        for label in ax.get_yticklabels():
            label.set_fontsize(5)

    fig.set_size_inches(11,8)
    for ext in ['png', 'pdf']:
        fn = "skyspec.%s" % (ext)
        fig.savefig(fn, dpi=150, bbox_inches='tight')
        print("done with %s" % (ext))



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




def plot_sky_spectrum(wl, flux,
                      good_sky_data=None, bad_rows=None,
                      n_rows=9,
                      output_filebase="skyspec",
                      plot_every=25,
                      sky_spline=None,
                      ext_list=None,
                      skylines=None,
                      basepoints=None,
                      ):

    logger = logging.getLogger("PlotSkySpec")

    if (ext_list is None):
        ext_list = ['png'] #, 'pdf']
    if (ext_list is not None and type(ext_list) != list):
        ext_list = list(ext_list)

    wl_bad = None
    flux_bad = None

    if (good_sky_data is not None):
        #good_sky_data = hdulist['GOOD_SKY_DATA'].data.astype(numpy.bool)
        _flux = flux.copy()
        _wl = wl.copy()
        flux = flux[good_sky_data]
        wl = wl[good_sky_data]

        wl_bad = _wl[~good_sky_data]
        flux_bad = _flux[~good_sky_data]

        valid = numpy.isfinite(wl_bad) & numpy.isfinite(flux_bad)
        wl_bad = wl_bad[valid]
        flux_bad = flux_bad[valid]
        logger.info("Using only GOOD_SKY_DATA for plotting")

    elif (bad_rows is not None):

        flux = flux[~bad_rows]
        wl = wl[~bad_rows]
        logger.info("Using only data from rows not in BADROWS")

    else:
        logger.info("Using all pixels")
        pass
            # badrows = numpy.ones_like(flux, dtype=numpy.bool)

    good_data = numpy.isfinite(flux)
    flux = flux[good_data]
    wl = wl[good_data]

    fig = pyplot.figure()
    #n_rows = 9

    wl_min = numpy.min(wl)
    wl_max = numpy.max(wl)
    logger.debug("Wavelength range: %f -- %f" % (wl_min, wl_max))
    overlap = 0.02 * (wl_max-wl_min) / n_rows

    every = plot_every
    everyfit = every**2

    # si = numpy.argsort(wl)
    # sky_spline = scipy.interpolate.UnivariateSpline(
    #     x=wl[si].flatten()[::everyfit], y=flux[si].flatten()[::everyfit],
    # )

    for row in range(n_rows):
        ax = fig.add_subplot(n_rows, 1, row+1)

        this_wl_min = float(row)/n_rows * (wl_max-wl_min) + wl_min - overlap
        this_wl_max = float(row+1)/n_rows * (wl_max-wl_min) + wl_min + overlap
        logger.debug("row %2d ==> %.1f -- %.1f" % (
            row, this_wl_min, this_wl_max)
        )

        if (wl_bad is not None and flux_bad is not None):
            in_wl_range = (wl_bad >= this_wl_min) & (wl_bad <= this_wl_max)
            this_wl = wl_bad[in_wl_range][::every]
            this_flux = flux_bad[in_wl_range][::every]
            ax.plot(this_wl, this_flux, marker=",", color="#a0a0a0",
                    linestyle='None')


        in_wl_range = (wl >= this_wl_min) & (wl <= this_wl_max)
        this_wl = wl[in_wl_range][::every]
        this_flux = flux[in_wl_range][::every]
        # print row, this_wl.shape


        ax.plot(this_wl, this_flux, "b,")
        ax.set_xlim((this_wl_min, this_wl_max))
        y_max = 500

        #print sky_spline
        if (sky_spline is not None):
            highres_wl = numpy.linspace(this_wl_min, this_wl_max, 1000)
            #print highres_wl
            highres_flux = sky_spline(highres_wl)
            ax.plot(highres_wl, highres_flux, 'r-')
            numpy.savetxt("specblock.%d" % (row+1),
                          numpy.array([this_wl, this_flux]).T)
            numpy.savetxt("specblock_fit.%d" % (row+1),
                          numpy.array([highres_wl, highres_flux]).T)
            max_spline_flux = numpy.max(highres_flux)
            y_max = 1.1*max_spline_flux

            if (basepoints is not None):
                basepoint_values = sky_spline(basepoints)
                ax.plot(basepoints, basepoint_values,
                        color='red', marker='o', markersize=4,
                        fillstyle='full',
                        markeredgecolor='red',
                        markeredgewidth=0.0,
                        linewidth=0)


        ax.set_ylim((0,y_max))

        #
        # Draw vertical markers where we found sky-lines
        #
        if (skylines is not None):
            skylines_here_select = (skylines[:,-1] > this_wl_min) & \
                                   (skylines[:,-1] < this_wl_max)
            skylines_here = skylines[skylines_here_select]
            for sl in skylines_here:
                sky_wl = sl[-1]
                ax.annotate("", xy=(sky_wl,0.7*y_max), xytext=(sky_wl,0),
                            arrowprops=dict(facecolor='black', shrink=0.0,
                                            width=0.5, frac=0.1, headwidth=4),
                )
                #print("skyline at %f" % (sky_wl))

        for label in ax.get_xticklabels():
            label.set_fontsize(5)
            x,y = label.get_position()
            # print x,y
            label.set_position((x,y+0.07))
            #label.set_verticalalignment('top')
        for label in ax.get_yticklabels():
            label.set_fontsize(5)

    fig.set_size_inches(11,8)

    for ext in ext_list:
        fn = "%s.%s" % (output_filebase, ext)
        logger.debug("Starting to create %s" % (fn))
        fig.savefig(fn, dpi=150, bbox_inches='tight')
        logger.info("done writing %s" % (fn))



if __name__ == "__main__":

    logsetup = pysalt.mp_logging.setup_logging()

    fn = sys.argv[1]

    #allskies = numpy.loadtxt(fn)
    #print allskies.shape


    hdulist = fits.open(fn)
    flux = hdulist['SCI'].data
    wl = hdulist['WAVELENGTH'].data

    try:
        good_sky_data = hdulist['GOOD_SKY_DATA'].data.astype(numpy.bool)
    except:
        good_sky_data = None

    try:
        bad_rows = hdulist['BADROWS'].data > 0
    except:
        bad_rows = None

    try:
        skytable_ext = hdulist['SKYLINES']
        n_lines = skytable_ext.header['NAXIS2']
        n_cols = skytable_ext.header['TFIELDS']
        skyline_list = numpy.empty((n_lines, n_cols))
        for i in range(n_cols):
            skyline_list[:,i] = skytable_ext.data.field(i)
        print skyline_list
        wl_poly = numpy.zeros((4))
        skyline_list[:,-1] = 0.
        for i in range(4):
            fac = hdulist[0].header['WLSFIT_%d' % (i)]
            skyline_list[:,-1] += fac * numpy.power(skyline_list[:,0],i)
        #skyline_list[:,-1] = numpy.polyval(wl_poly, skyline_list[:,0])
        print skyline_list[:,-1]
    except Exception as e:
        print e
        skyline_list = None

    plot_sky_spectrum(
        wl=wl, flux=flux,
        good_sky_data=good_sky_data,
        bad_rows=bad_rows,
        n_rows=9,
        output_filebase=fn[:-5]+".skyspec",
        plot_every=25,
        sky_spline=None,
        ext_list=None,
        skylines=skyline_list,
    )

    pysalt.mp_logging.shutdown_logging(logsetup)

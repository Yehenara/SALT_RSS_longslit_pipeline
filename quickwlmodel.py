#!/bin/env python


import os
import sys
import numpy
from astropy.io import fits

import logging
import scipy
import scipy.interpolate
import time

import find_sources
import tracespec

import pysalt.mp_logging


def find_additional_basepoints(data, d_step=4, presmooth=2,
                                   debug=False):

    logger = logging.getLogger("FindAddtlBasepoints")

    wl = data[:,0]
    _flux = data[:,2].copy()

    logger.debug("Input: %d sky-samples" % (wl.shape[0]))
    logger.debug("gaussian pre-smooth: %.1f pixels" % (presmooth))
    logger.debug("slope delta-step: %d intervals" % (d_step))

    smoothed = numpy.pad(
        scipy.ndimage.filters.gaussian_filter(
            input=_flux, sigma=presmooth,
            order=0, output=None,
            mode='constant', cval=0.0,
        ), pad_width=2, mode='constant',
    )

    #_delta_flux = _flux[d_step:] - _flux[:-d_step]
    delta_flux = smoothed[d_step:] - smoothed[:-d_step]
    _delta_flux = delta_flux.copy()
    for iter in range(3):

        _stats = numpy.nanpercentile(_delta_flux, [16, 50, 84])
        one_sigma = 0.5 * (_stats[2] - _stats[0])
        median = _stats[1]
        bad = (_delta_flux > median + 3 * one_sigma) | \
              (_delta_flux < median - 3 * one_sigma)
        _delta_flux[bad] = numpy.NaN
        # print "delta-flux:", iter, median, one_sigma
        logger.debug("Iteration %d: typical gradients: %f +/- %f" % (
            iter, median, one_sigma
        ))

    significant = numpy.fabs(delta_flux) > 3*one_sigma
    adtl_basepoints = wl[significant]

    if (debug):
        numpy.savetxt("quickspec.deltaflux.trim", _delta_flux)
        numpy.savetxt("quickspec.deltaflux", delta_flux)
        numpy.savetxt("quickspec.basepoints", adtl_basepoints)

    logger.debug("Found %d points of significant slope" % (
        adtl_basepoints.shape[0])
    )

    return adtl_basepoints



def find_additional_basepoints_old(data, bs=5):

    # now reshape the input array and try to find regions where the
    # flux-level changes rapidly
    print data.shape
    # binsize = 10
    # n_points = data.shape[0] % binsize
    # good_data = data[:-n_points]
    # print good_data.shape

    local_var = numpy.empty((data.shape[0]))
    local_var[:] = numpy.NaN
    for i in range(data.shape[0]):
        local_var[i] = numpy.var(data[i-bs:i+bs+1, 1])

    numpy.savetxt("quickspec.localvar",
                  numpy.array([data[:,0], local_var]).T)

    _var = local_var.copy()
    for iter in range(3):
        _stats = numpy.nanpercentile(_var, [16, 50, 84])
        one_sigma = 0.5 * (_stats[2] - _stats[0])
        median = _stats[1]
        bad = (_var > median+3*one_sigma) | (_var < median-3*one_sigma)
        _var[bad] = numpy.NaN
        print "localvar", iter, median, one_sigma

    strong_gradient = numpy.isfinite(local_var)  & \
                      (local_var > median+3*one_sigma)

    basepoints = data[:,0][strong_gradient]

    return basepoints


if __name__ == "__main__":

    logsetup = pysalt.mp_logging.setup_logging()

    fn = sys.argv[1]

    data = numpy.loadtxt(fn)

    good = numpy.isfinite(data[:,0]) & numpy.isfinite(data[:,1]) & \
           numpy.isfinite(data[:,2])
    data = data[good]

    wl = data[:,0]
    flux = data[:,2]
    noise = data[:,1]

    wl_min, wl_max = numpy.min(wl), numpy.max(wl)

    spline_iter = scipy.interpolate.LSQUnivariateSpline(
        x=wl,
        y=flux,
        t=wl[1:-1][::3],
        # t=k_iter_good, #k_wl,
        w=noise, # no weights (for now)
        bbox=[wl_min, wl_max],
        k=3, # use a cubic spline fit
    )

    highres_x = numpy.linspace(wl_min, wl_max, 25000)
    highres_flux = spline_iter(highres_x)
    numpy.savetxt("quickspec.txt",
                  numpy.array([highres_x, highres_flux]).T)

    print highres_flux


    adtl_basepoints = find_additional_basepoints(
        data=data,
        debug=True,
    )
    print "Found %d new basepoints" % (adtl_basepoints.shape[0])



    # now reshape the input array and try to find regions where the
    # flux-level changes rapidly
    print data.shape
    binsize = 10
    n_points = data.shape[0] % binsize
    good_data = data[:-n_points]
    print good_data.shape

    local_var = numpy.empty((data.shape[0]))
    local_var[:] = numpy.NaN
    bs = 5
    for i in range(data.shape[0]):
        local_var[i] = numpy.var(data[i-bs:i+bs+1, 1])

    numpy.savetxt("quickspec.localvar",
                  numpy.array([data[:,0], local_var]).T)

    _var = local_var.copy()
    for iter in range(3):
        _stats = numpy.nanpercentile(_var, [16, 50, 84])
        one_sigma = 0.5 * (_stats[2] - _stats[0])
        median = _stats[1]
        bad = (_var > median+3*one_sigma) | (_var < median-3*one_sigma)
        _var[bad] = numpy.NaN
        print "localvar", iter, median, one_sigma


    good_reshaped = good_data.reshape((-1, binsize, data.shape[1]))
    print good_reshaped.shape
    edge_var = numpy.var(good_reshaped, axis=1)
    edge_median = numpy.mean(good_reshaped, axis=1)
    print edge_var.shape, edge_median.shape
    numpy.savetxt("quickspec.X",
                 numpy.array([edge_median[:,0], edge_var[:,1]]).T)

    valid_var = edge_var[:,1].copy()
    for i in range(3):
        _stats = numpy.nanpercentile(valid_var, [16,50,84])
        one_sigma = 0.5*(_stats[2]-_stats[0])
        median = _stats[1]

        print i, median, one_sigma

        bad = (valid_var > (median+3*one_sigma)) | (valid_var < (
            median-3*one_sigma))
        valid_var[bad] = numpy.NaN

        numpy.savetxt("quickspec.X.%d" % (i+1),
                 numpy.array([edge_median[:,0], valid_var]).T)



    pysalt.mp_logging.shutdown_logging(logsetup)

    os._exit(0)


    # fits_fn = sys.argv[2]
    #
    # hdulist = fits.open(fits_fn)
    # obj_wl = hdulist['WAVELENGTH'].data
    # padded = numpy.empty((obj_wl.shape[0], obj_wl.shape[1] + 2))
    # padded[:, 1:-1] = obj_wl[:, :]
    # padded[:, 0] = obj_wl[:, 0]
    # padded[:, -1] = obj_wl[:, -1]
    # from_wl = 0.5 * (padded[:, 0:-2] + padded[:, 1:-1])
    # to_wl = 0.5 * (padded[:, 1:-1] + padded[:, 2:])
    #
    # print("computing full-res sky frame")
    # t0 = time.time()
    # sky2d = numpy.array([spline_iter.integral(a, b) for a, b in
    #                      zip(from_wl.ravel(), to_wl.ravel())]).reshape(
    #     obj_wl.shape)
    # print("done after %f seconds" % (time.time()-t0))
    #
    #
    # fits.PrimaryHDU(data=sky2d).writeto("quicksky.fits", clobber=True)
    # print("all done!")

#!/usr/bin/env python


import os, sys, numpy, scipy, scipy.interpolate
import spline_pickle_test
from astropy.io import fits

if __name__ == "__main__":

    fn = sys.argv[1]

    splinein_fn = fn+"__splinein"
    spline = spline_pickle_test.read_pickle(splinein_fn)

    print spline


    wl_center = "OBJ_"+fn+".fits"
    wl_from_fn = fn+"__IntegSky_from.fits"
    wl_to_fn = fn+"__IntegSky_to.fits"

    obj_hdu = fits.open(wl_center)
    wl_from = fits.open(wl_from_fn)[0].data
    wl_to = fits.open(wl_to_fn)[0].data
    wl_center = obj_hdu['WAVELENGTH'].data
    wl_width = wl_to - wl_from

    print wl_from.shape, wl_to.shape, wl_center.shape

    ycol = wl_center.shape[0]/2
    lookup1d = spline(wl_center[ycol,:]).reshape((-1,1))
    integ1d = numpy.array([spline.integral(a, b) for a, b in zip(wl_from[ycol,:], wl_to[ycol,:])]).reshape((-1,1))

    combined = numpy.empty((wl_center.shape[1], 4))
    combined[:,0] = wl_center[ycol,:]
    combined[:,1] = lookup1d[:,0]
    combined[:,2] = integ1d[:,0]
    combined[:,3] = wl_width[ycol,:]

    print lookup1d.shape, integ1d.shape
    numpy.savetxt(fn+"__fiddle_splinecomp", combined)

    print "computing full 2-d sky"
    #
    # full 2d
    #
    sky2d = numpy.array([spline.integral(a, b) for a, b in zip(wl_from.ravel(), wl_to.ravel())]).reshape(
        wl_from.shape)
    sky2d /= wl_width
    print "2d-sky:", sky2d.shape

    img = obj_hdu['SCI.RAW'].data
    skysub = img - sky2d

    out_fn = fn+"__fiddle_skysub.fits"
    fits.PrimaryHDU(header=obj_hdu['SCI'].header,
                        data=skysub).writeto(out_fn, clobber=True)
    print "skysub written to",out_fn
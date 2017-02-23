#!/usr/bin/env python

import os
import sys
import astropy.io.fits as fits
import numpy
import math
import itertools
import scipy
import scipy.interpolate
import logging
import time

import find_sources
import tracespec
import optimal_extraction


import pysalt.mp_logging
from optparse import OptionParser





def rectify_full_spec(
        data, var, wavelength,
        traceoffset=None,
        biny=1, spec_resolution=None,
        debug=False):

    logger = logging.getLogger("RectifyFullSpec")

    #
    # Set default dispersion to mean dispersion
    #
    wl_min = numpy.min(wavelength)
    wl_max = numpy.max(wavelength)
    if (spec_resolution is None):
        spec_resolution = (wl_max - wl_min) / wavelength.shape[1]
        logger.debug("average dispersion: %f A/px" % (spec_resolution))
    logger.debug("using spectral resolution of %f" % (spec_resolution))

    #
    # Determine the position input grid, accounting for position offsets
    # determined from the slit/source trace
    #
    pos_y, pos_x = numpy.indices(data.shape, dtype=numpy.float)
    if (traceoffset is not None):
        pos_y -= traceoffset


    #
    # Compute the coordinates of the output grid
    #
    y_min = numpy.min(pos_y)
    y_max = numpy.max(pos_y)

    n_samples = int((y_max - y_min) / biny)

    y_start = (numpy.arange(n_samples, dtype=numpy.float)*biny + y_min)
    y_width = biny
    y_end = y_start + y_width

    #
    #
    #
    wl_pad = numpy.pad(wavelength, ((0,0),(1,1)), mode='edge')
    wl_start = 0.5*(wl_pad[:, 0:-2] + wl_pad[:, 1:-1])
    wl_end = 0.5*(wl_pad[:, 1:-1] + wl_pad[:, 2:])

    #
    # select only good & valid data
    #
    good_and_valid = numpy.isfinite(data)

    use_data = data[good_and_valid]
    use_var = var[good_and_valid]
    use_wl = wavelength[good_and_valid]
    use_wl_start = wl_start[good_and_valid]
    use_wl_end = wl_end[good_and_valid]
    use_pos_y = pos_y[good_and_valid]
    use_pos_x = pos_x[good_and_valid]

    # #
    # # average dispersion
    # #
    # avg_dispersion = numpy.mean(profile2d[:, 6] - profile2d[:, 5])
    # spec_resolution = wl_resolution if wl_resolution > 0 else avg_dispersion * math.fabs(wl_resolution)
    # logger.debug("using spectral resolution of %f" %  (spec_resolution))

    #
    # Now drizzle the data into the 2-d wavelength/y data buffer
    #
    # wl_min = numpy.min(profile2d[:,5])
    # wl_max = numpy.max(profile2d[:,6])
    n_spec_bins = int(math.ceil((wl_max - wl_min) / spec_resolution))
    logger.debug("using a total of %d spectral bins" % (n_spec_bins))

    out_drizzle = numpy.zeros((n_samples, n_spec_bins))
    out_drizzle_var = numpy.zeros((n_samples, n_spec_bins))
    print out_drizzle.shape

    print use_data.shape
    #return


    wl_start = numpy.arange(n_spec_bins, dtype=numpy.float) * spec_resolution + wl_min
    wl_end = wl_start + spec_resolution

    start_time = time.time()
    for px in range(use_data.shape[0]):
        # px = [ pixel / var / x / wavelength / y / wl1 / wl2]
        # [flux, var, x, wl, y, wl1, wl2] = px

        if (px > 0 and (px%10000) == 0):
            used_time = time.time() - start_time
            time_to_complete = (use_data.shape[0]-px)*used_time/px
            sys.stdout.write("\r%7d of %7d pixels --> %5.1f%% complete, %5.1fs left" % (
                px,use_data.shape[0], 100.*float(px)/float(use_data.shape[0]), time_to_complete))
            sys.stdout.flush()

        flux = use_data[px]
        var = use_var[px]
        x = use_pos_x[px]
        wl = use_wl[px]
        y = use_pos_y[px]
        wl1 = use_wl_start[px]
        wl2 = use_wl_end[px]


        y1 = y - 0.5
        y2 = y + 0.5

        first_y = (y1 - y_min) / y_width #* supersample
        last_y = (y2 - y_min) / y_width #* supersample

        _y1 = int(math.floor(first_y))
        _y2 = int(math.ceil(last_y))
        _wl1 = int(math.floor((wl1 - wl_min)/spec_resolution))
        _wl2 = int(math.ceil((wl2 - wl_min)/spec_resolution))

        #print px
        #print y1, y2, wl1, wl2, "-->", _y1, _y2, _wl1, _wl2, "(", y_min, wl_min, supersample, wl_resolution,")"

        # Determine drizzle=-factors along the spatial axis
        pixels_y = range(_y1, _y2)
        pixel_fraction_y = numpy.zeros((_y2-_y1,1))
        for i, tp in enumerate(pixels_y):
            #print "drizzle pixel",i,tp
            if (i == 0 and i == len(pixels_y) - 1):
                # all flux in a single output pixel
                fraction = 1.0
            elif (i == 0):
                # right-edge of this pixel is
                right_edge = (tp+1)*y_width + y_min
                #print "right-edge:", right_edge
                f = right_edge - y1
                # first pixel, with more to come
                fraction = f/(y2-y1)
            elif (i == len(pixels_y) - 1):
                # last of many pixels
                left_edge = tp*y_width + y_min
                #print "left edge", left_edge
                f = y2-left_edge
                fraction = f/(y2-y1)
            else:
                # some pixel in the middle
                fraction = y_width / (y2-y1)
            #print fraction
            pixel_fraction_y[i,0] = fraction



        #print pixel_fraction_y, numpy.sum(pixel_fraction_y)

        # Now also figure out the drizzle-factor along the wavelength axis
        pixels_wl = range(_wl1, _wl2)
        pixel_fraction_wl = numpy.zeros((1, _wl2-_wl1))
        for i, tp in enumerate(pixels_wl):
            #print "drizzle pixel",i,tp
            if (i == 0 and i == len(pixels_wl) - 1):
                # all flux in a single output pixel
                fraction = 1.0
            elif (i == 0):
                # right-edge of this pixel is
                right_edge = (tp + 1) * spec_resolution + wl_min
                #print "right-edge:", right_edge
                f = right_edge - wl1
                # first pixel, with more to come
                fraction = f / (wl2 - wl1)
            elif (i == len(pixels_wl) - 1):
                # last of many pixels
                left_edge = tp * spec_resolution + wl_min
                #print "left edge", left_edge
                f = wl2 - left_edge
                fraction = f / (wl2 - wl1)
            else:
                # some pixel in the middle
                fraction = spec_resolution / (wl2 - wl1)

            #print fraction
            pixel_fraction_wl[0, i] = fraction

        #print pixel_fraction_wl, numpy.sum(pixel_fraction_wl)

        full_weight_matrix = pixel_fraction_wl * pixel_fraction_y
        #print full_weight_matrix.shape
        #print full_weight_matrix
        #print numpy.sum(full_weight_matrix)

        flux_distribution = full_weight_matrix * flux
        var_distribution = full_weight_matrix * var

        try:
            out_drizzle[_y1:_y2, _wl1:_wl2] += flux_distribution
            out_drizzle_var[_y1:_y2, _wl1:_wl2] += var_distribution
        except ValueError:
            pass

        #if (_XXX > 20):
        #    break

    #
    # truncate negative pixels to 0
    #
    if (debug):
        fits.PrimaryHDU(data=out_drizzle).writeto("out_drizzle0.fits", clobber=True)
        #out_drizzle[out_drizzle<0] = 0.

        # y,x = numpy.indices(out_drizzle.shape)
        # combined = numpy.empty((x.shape[0]*x.shape[1], 3))
        # combined[:,0] = x.flatten()
        # combined[:,1] = y.flatten()
        # combined[:,2] = out_drizzle.flatten()
        # numpy.savetxt("drizzled", combined)
        # numpy.savetxt("drizzled2", out_drizzle)
        # fits.PrimaryHDU(data=out_drizzle).writeto("out_drizzle.fits", clobber=True)

    hdr = fits.Header()
    hdr['WCSNAME'] = "wavelength"

    hdr['CRPIX1'] = 1.0
    hdr['CRPIX2'] = 1.0

    hdr['CRVAL1'] = wl_min
    hdr['CTYPE1'] = 'AWAV'
    hdr['CUNIT1'] = "Angstrom"
    hdr['CD1_1'] = spec_resolution

    hdr['CRVAL2'] = 1.
    hdr['CTYPE2'] = 'POS'
    hdr['CUNIT2'] = 'pixel'
    hdr['CD2_2'] = 1.0

    rect_flux = fits.ImageHDU(data=out_drizzle, name="SCI.RECT", header=hdr)
    rect_var = fits.ImageHDU(data=out_drizzle_var, name='VAR.RECT', header=hdr)

    return rect_flux, rect_var


    #
    # Now integrate the drizzle spectrum along the slit to compute the relative contribution of each pixel.
    #
    spec1d = numpy.sum(out_drizzle, axis=0)
    if (debug):
        print out_drizzle.shape, spec1d.shape
        numpy.savetxt("spec1d", numpy.append(wl_start.reshape((-1,1)), spec1d.reshape((-1,1)), axis=1))
        numpy.savetxt("optprofile1d", numpy.sum(out_drizzle, axis=1))
        numpy.savetxt("optprofile1d.mean", numpy.mean(out_drizzle, axis=1))

    median_flux = numpy.median(spec1d)
    logger.debug("median flux level: %f" % (median_flux))
    bad_data = spec1d < 0.05*median_flux
    spec1d[bad_data] = numpy.NaN

    drizzled_weight = out_drizzle / spec1d.reshape((1, -1))

    # now we have bad columns set to NaN, this makes interpolation tricky
    # therefore we fill in these areas with the global profile
    no_data = numpy.isnan(drizzled_weight)
    global_profile = numpy.sum(out_drizzle, axis=1)
    global_profile /= numpy.sum(global_profile)
    global_profile_2d = numpy.repeat(global_profile.reshape((-1, 1)), spec1d.shape[0], axis = 1)
    # print "global profile:", global_profile_2d.shape
    drizzled_weight[no_data] = global_profile_2d[no_data]

    if (debug):
        combined[:,2] = drizzled_weight.flatten()
        numpy.savetxt("drizzled.norm", combined)
        fits.PrimaryHDU(data=drizzled_weight).writeto("drizzled-weight.fits", clobber=True)
        #combined[:,2] = (drizzled_weight / numpy.sum(drizzled_weight, axis=0).reshape((1,-1))).flatten()
        #numpy.savetxt("drizzled.normsum", combined)

        numpy.savetxt("drizzled.1d", numpy.mean(drizzled_weight, axis=1))

    #
    # Finally, limit all negative pixels to 0
    #
    drizzled_weight[drizzled_weight<0] = 0.
    if (debug):
        numpy.savetxt("drizzled.1dv2",
                      numpy.mean(drizzled_weight, axis=1))

    #
    # Now we have a full 2-d distribution of extraction weights as fct. of dy and wavelength
    #
    #drizzled_weight[drizzled_weight<0] = 0
    logger.debug("computing final 2-d interpolator")
    opt_weight = OptimalWeight(
        wl_min=wl_min, wl_step=spec_resolution,
        y_min=y_min, y_step=y_width,
        data=drizzled_weight,
    )

    # weight_interpol = scipy.interpolate.interp2d(
    #     x=combined[:,0],
    #     y=combined[:,1],
    #     z=combined[:,2],
    #     kind='linear',
    #     copy=True,
    #     bounds_error=False,
    #     fill_value=0.,
    # )
    return opt_weight





if __name__ == "__main__":


    logger = pysalt.mp_logging.setup_logging()

    parser = OptionParser()
    parser.add_option("", "--dwl", dest="dwl",
                      help="output dispersion",
                      default=None, type=int)
    parser.add_option("", "--ybin", dest="ybin",
                      help="binning in y-direction (spatial)",
                      default=2, type=int)
    parser.add_option("", "--tracex", dest="tracex",
                      default=None, type=int)
    parser.add_option("", "--tracey", dest="tracey",
                      default=None, type=int)
    # parser.add_option("", "--ycenter", dest="ycenter",
    #                   help="y center position in un-binned pixels",
    #                   default=2170, type=float)

    (options, cmdline_args) = parser.parse_args()


    fits_fn = cmdline_args[0]
    fits_out = cmdline_args[1]
    print "%s --> %s" % (fits_fn, fits_out)

    hdulist = fits.open(fits_fn)

    if (options.tracex is not None and
        options.tracey is not None):

        # run a trace

        print "computing spectrum trace"
        spectrace_data = tracespec.compute_spectrum_trace(
            data=hdulist['SKYSUB.OPT'].data,
            start_x=options.tracex, start_y=options.tracey,
            xbin=1,
            debug=True
        )
        print "finding trace slopes"
        slopes, traceoffset = tracespec.compute_trace_slopes(spectrace_data)

    else:
        try:
            traceoffset = hdulist['TRACEOFFSET'].data
        except:
            traceoffset = None

    # default trace-offset: None
    rect_data, rect_var = rectify_full_spec(
        data=hdulist['SKYSUB.OPT'].data,
        var=hdulist['VAR'].data,
        wavelength=hdulist['WAVELENGTH'].data,
        traceoffset=traceoffset,
        biny=options.ybin,
        spec_resolution=options.dwl,
        debug=True,
    )
    print

    print "All done!"

    out_hdulist = fits.HDUList([
        hdulist[0],
        rect_data, rect_var,
    ])
    out_hdulist.writeto(fits_out, clobber=True)

    pysalt.mp_logging.shutdown_logging(logger)


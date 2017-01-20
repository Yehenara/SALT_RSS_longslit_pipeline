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

import find_sources
import tracespec

class OptimalWeight(object):

    def __init__(self, wl_min, wl_step, y_min, y_step, data):
        self.wl_min = wl_min
        self.wl_step = wl_step
        self.y_min = y_min
        self.y_step = y_step
        self.data = data

        self.prepare()

    def prepare(self):
        self.y_center = numpy.arange(self.data.shape[0], dtype=numpy.float) * self.y_step + 0.5*self.y_step + self.y_min
        self.wl_center = numpy.arange(self.data.shape[1], dtype=numpy.float) * self.wl_step + 0.5*self.wl_step + self.wl_min

        print "Y/wl:", self.y_center.shape, self.wl_center.shape

        self.profile1d = numpy.mean(self.data, axis=1)

        self.weight_interpol = scipy.interpolate.RectBivariateSpline(
            x=self.y_center,
            y=self.wl_center,
            z=self.data,
            kx=1, ky=1,
        )
        print self.weight_interpol

    def get_weight(self, wl, y, mode='interpolate'):
        #print wl, y
        wl = numpy.array(wl)
        y = numpy.array(y)

        # find look-up indices for wl and y
        # make sure they are in the valid range
        i_wl = numpy.array((wl - self.wl_min) / self.wl_step, dtype=numpy.int)
        i_wl[i_wl < 0] = 0
        i_wl[i_wl >= self.data.shape[1]] = self.data.shape[1]-1

        i_y = numpy.array((y - self.y_min) / self.y_step, dtype=numpy.int)
        i_y[i_y < 0] = 0
        i_y[i_y >= self.data.shape[0]] = self.data.shape[0]-1

        #print i_wl,  i_y

        weight = self.data[i_y, i_wl]
        #print weight


        # if (mode == 'interpolate'):
        #     # find the closest
        #     return self.weight_interpol(wl,y)
        # else:
        #     # pick the closest point based on wl and y
        #     pass
        #
        # print "optimal weight @", wl_data_1d[px], y_data_1d[px], " == ", opt_weight
        return weight

    def get_normalization(self, y1, y2):

        in_range = (self.y_center > y1) & (self.y_center < y2)
        cutout_1d = self.profile1d[in_range]

        return numpy.sum(cutout_1d)





def generate_source_profile(data, variance, wavelength, trace_offset,
                            position, width=50):

    # get coordinates
    y,x = numpy.indices(data.shape, dtype=numpy.float)

    pos_x, pos_y = position[0], position[1]

    print y.shape
    print trace_offset.shape

    traceoffset2d = numpy.repeat(trace_offset.reshape((1,-1)), data.shape[0], axis=0)
    print traceoffset2d.shape

    dy = y - pos_y - traceoffset2d
    fits.PrimaryHDU(data=dy).writeto("optex_dy.fits", clobber=True)

    wl_pad = numpy.pad(wavelength, ((0,0),(1,1)), mode='edge')
    wl_start = 0.5*(wl_pad[:, 0:-2] + wl_pad[:, 1:-1])
    wl_end = 0.5*(wl_pad[:, 1:-1] + wl_pad[:, 2:])
    print "wl start/end", wl_start.shape, wl_end.shape

    in_range = numpy.fabs(dy) < width+0.5
    src_data = data[in_range]
    src_var = variance[in_range]
    src_x = x[in_range]
    src_y = dy[in_range]
    src_wavelength = wavelength[in_range]
    src_wl1 = wl_start[in_range]
    src_wl2 = wl_end[in_range]

    combined = numpy.empty((src_data.shape[0], 7))
    combined[:,0] = src_data
    combined[:,1] = src_var
    combined[:,2] = src_x
    combined[:,3] = src_wavelength
    combined[:,4] = src_y
    combined[:,5] = src_wl1
    combined[:,6] = src_wl2
    numpy.savetxt("optex.data", combined)

    return combined


def integrate_source_profile(width=25, supersample=10, wl_resolution=5, profile2d=None):


    if (profile2d is None):
        print "no data given, nothing to do"
        return None

    print "running integrate_source_profile"

    #
    # now drizzle the data into the output container
    #
    y_min = -width - 0.5
    y_max = width + 0.5

    n_samples = (y_max - y_min) * supersample
    y_start = (numpy.arange(n_samples, dtype=numpy.float)/supersample - width)
    y_width = 1./supersample
    y_end = y_start + y_width

    # print y_start
    # print y_end

    #
    # select only good & valid data
    #
    good_and_valid = numpy.isfinite(profile2d[:,0])
    profile2d = profile2d[good_and_valid]

    # #
    # # Now convert the profile data into a super-sampled 2-d array in y/wl
    # # interpolate where necessary
    # #
    # print profile2d.shape, profile2d.shape[0]/width, width
    # print "Preparing profile interpolation and integration"
    # interpol = scipy.interpolate.interp2d(
    #     x=profile2d[:,3],
    #     y=profile2d[:,4],
    #     z=profile2d[:,0],
    #     bounds_error=False, fill_value=0, kind='linear',
    # )

    #
    # average dispersion
    #
    avg_dispersion = numpy.mean(profile2d[:, 6] - profile2d[:, 5])
    print "average dispersion:", avg_dispersion
    spec_resolution = wl_resolution if wl_resolution > 0 else avg_dispersion * math.fabs(wl_resolution)
    print "using spectral resolution of", spec_resolution

    #
    # Now drizzle the data into the 2-d wavelength/y data buffer
    #
    wl_min = numpy.min(profile2d[:,5])
    wl_max = numpy.max(profile2d[:,6])
    n_spec_bins = int(math.ceil((wl_max - wl_min) / spec_resolution))
    print "using a total of %d spectral bins" % (n_spec_bins)

    out_drizzle = numpy.zeros((n_samples, n_spec_bins))
    print out_drizzle.shape

    wl_start = numpy.arange(n_spec_bins, dtype=numpy.float) * spec_resolution + wl_min
    wl_end = wl_start + spec_resolution
    for _XXX, px in enumerate(profile2d):
        # px = [ pixel / var / x / wavelength / y / wl1 / wl2]
        [flux, var, x, wl, y, wl1, wl2] = px

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

        try:
            out_drizzle[_y1:_y2, _wl1:_wl2] += flux_distribution
        except ValueError:
            pass

        #if (_XXX > 20):
        #    break

    #
    # truncate negative pixels to 0
    #
    fits.PrimaryHDU(data=out_drizzle).writeto("out_drizzle0.fits", clobber=True)
    #out_drizzle[out_drizzle<0] = 0.

    y,x = numpy.indices(out_drizzle.shape)
    combined = numpy.empty((x.shape[0]*x.shape[1], 3))
    combined[:,0] = x.flatten()
    combined[:,1] = y.flatten()
    combined[:,2] = out_drizzle.flatten()
    numpy.savetxt("drizzled", combined)
    numpy.savetxt("drizzled2", out_drizzle)
    fits.PrimaryHDU(data=out_drizzle).writeto("out_drizzle.fits", clobber=True)

    #
    # Now integrate the drizzle spectrum along the slit to compute the relative contribution of each pixel.
    #
    spec1d = numpy.sum(out_drizzle, axis=0)
    print out_drizzle.shape, spec1d.shape
    numpy.savetxt("spec1d", numpy.append(wl_start.reshape((-1,1)), spec1d.reshape((-1,1)), axis=1))
    numpy.savetxt("optprofile1d", numpy.sum(out_drizzle, axis=1))
    numpy.savetxt("optprofile1d.mean", numpy.mean(out_drizzle, axis=1))

    median_flux = numpy.median(spec1d)
    print "median flux level:", median_flux
    bad_data = spec1d < 0.05*median_flux
    spec1d[bad_data] = numpy.NaN

    drizzled_weight = out_drizzle / spec1d.reshape((1, -1))

    # now we have bad columns set to NaN, this makes interpolation tricky
    # therefore we fill in these areas with the global profile
    no_data = numpy.isnan(drizzled_weight)
    global_profile = numpy.sum(out_drizzle, axis=1)
    global_profile /= numpy.sum(global_profile)
    global_profile_2d = numpy.repeat(global_profile.reshape((-1, 1)), spec1d.shape[0], axis = 1)
    print "global profile:", global_profile_2d.shape
    drizzled_weight[no_data] = global_profile_2d[no_data]

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
    numpy.savetxt("drizzled.1dv2", numpy.mean(drizzled_weight, axis=1))

    #
    # Now we have a full 2-d distribution of extraction weights as fct. of dy and wavelength
    #
    #drizzled_weight[drizzled_weight<0] = 0
    print "computing final 2-d interpolator"
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





def optimal_extract(img_data, wl_data, variance_data,
                    trace_offset,
                    opt_weight_center_y=None,
                    optimal_weight=None,
                    minwl=-1, maxwl=-1, dwl=-1,
                    y_ranges=None,
                    reference_x=None, reference_y=None,
                    ):

    logger = logging.getLogger("OptimalDrizzleSpec")

    # load image data and wavelength map
    # img_data = hdu[input_ext].data
    # wl_data = hdu['WAVELENGTH'].data
    # variance_data = hdu['VAR'].data

    wl0 = minwl
    wlmax = maxwl
    if (minwl < 0):
        wl0 = numpy.min(wl_data)
    if (maxwl < 0):
        wlmax = numpy.max(wl_data)
    if (dwl < 0):
        _disp = numpy.diff(wl_data)
        dwl = 0.5 * numpy.min(_disp)
        if (dwl <= 0):
            dwl = 0.1

    #
    # determine a reference column, if not specified
    # this is needed as the tracing introduces a dependency between y and x
    #
    if (reference_x is None):
        reference_x = int(img_data.shape[1]/2)

    # compute an effective y array that accounts for line shifts in y-direction
    # (which is compensated with the line trace)
    #
    # This essentially sets the center of the extraction region !!!
    #
    y_raw,_ = numpy.indices(img_data.shape, dtype=numpy.float)
    if (reference_y is not None):
        y_raw -= reference_y

    dy0 = trace_offset[reference_x]
    corrected_y = y_raw + 1.0 - trace_offset + dy0 # add 1 pixel for FITS conformity
    logger.info("Trace offset at reference x=+%d: %f" % (reference_x, dy0))
    numpy.savetxt("corrected_y", corrected_y.ravel())

    # Now prepare the output array
    out_wl_count = int((wlmax - wl0) / dwl) + 1
    logger.info("output: %d wavelength points from %f to %f in steps of %f angstroems" % (
        out_wl_count, wl0, wlmax, dwl))
    out_wl = numpy.arange(out_wl_count, dtype=numpy.float) * dwl + wl0

    spectra_1d = numpy.empty((out_wl_count, len(y_ranges), 3))
    variance_1d = numpy.empty((out_wl_count, len(y_ranges), 3))

    # pad the entire wavelength array so we know the wavelength range of each pixel
    wl_data_padded = numpy.pad(wl_data, ((0, 0), (1, 1)), mode='edge')
    wl_width = 0.5 * (wl_data_padded[:, 2:] - wl_data_padded[:, :-2])
    wl_from = 0.5 * (wl_data_padded[:, 0:-2] + wl_data_padded[:, 1:-1])
    wl_to = 0.5 * (wl_data_padded[:, 1:-1] + wl_data_padded[:, 2:])

    for cur_yrange, _yrange in enumerate(y_ranges):

        y1 = _yrange[0]
        y2 = _yrange[1]
        logger.info("Extracting 1-D spectrum from columns %d --- %d" % (y1, y2))

        xxx = open("dump_%d-%d" % (y1, y2), "w")

        #
        # extract data
        #
        in_y_range = (corrected_y > (y1-0.5)) & (corrected_y <= (y2+0.5))
        n_pixels_in_spec = numpy.sum(in_y_range)
        if (n_pixels_in_spec <= 0):
            logger.error("Invalid Y-range: %f - %f, continuing with next "
                         "aperture" % (y1,y2))
            continue

        spec_img_data = img_data[in_y_range]
        spec_wl_data = wl_data[in_y_range]
        spec_var_data = variance_data[in_y_range]
        spec_wl_from = wl_from[in_y_range]
        spec_wl_to = wl_to[in_y_range]
        spec_wl_width = wl_width[in_y_range]
        logger.info("total # of input pixels: %d" % (spec_img_data.size))

        spec_y_data = corrected_y[in_y_range]
        numpy.savetxt("spec_y_data", spec_y_data)
        # if (opt_weight_center_y is not None):
        #     logger.info("correcting extraction center to match optimal extraction weights")
        #     spec_y_data -= opt_weight_center_y
        # numpy.savetxt("spec_y_data2", spec_y_data)

        weight_data = numpy.ones_like(img_data, dtype=numpy.float)
        if (optimal_weight is not None):
            logger.info("Computing optimal extraction weights for all input pixels")
            weight_data = optimal_weight.get_weight(
                wl=spec_wl_data, y=spec_y_data)
            c = numpy.empty((spec_wl_data.ravel().shape[0], 3))
            c[:,0] = spec_wl_data.ravel()
            c[:,1] = spec_y_data.ravel()
            c[:,2] = weight_data.ravel()
            numpy.savetxt("weight_data", c)

            optimal_normalization = optimal_weight.get_normalization(y1,y2)
            logger.info("Using relative normalization of %f" % (optimal_normalization))
        else:
            optimal_normalization = 1.0
        # print img_data.shape, wl_data.shape

        #
        # prepare the output flux array, and initialize with NaNs
        #
        out_flux = numpy.zeros_like(out_wl)
        out_flux[:] = numpy.NaN
        out_var = numpy.zeros_like(out_wl)
        out_var[:] = numpy.NaN
        out_weight = numpy.zeros_like(out_wl)
        out_weight2 = numpy.zeros_like(out_wl)
        # print out_wl

        #
        # prepare a padded WL array
        #
        # print wl_width.shape, wl_from.shape, wl_to.shape

        # compute the flux density array (i.e. flux per wavelength) by dividing the
        # flux image (which is flux integrated over the width of a pixel) by the width of each pixel
        img_data_per_wl = spec_img_data / spec_wl_width
        var_data_per_wl = spec_var_data / spec_wl_width

        #
        # now drizzle the data into the output container
        #
        wl_width_1d = spec_wl_width.ravel()
        wl_from_1d = spec_wl_from.ravel()
        wl_to_1d = spec_wl_to.ravel()
        wl_data_1d = spec_wl_data.ravel()
        img_data_per_wl_1d = img_data_per_wl.ravel()
        var_data_1d = spec_var_data.ravel()
        var_data_per_wl_1d = var_data_per_wl.ravel()
        wl_width_1d = spec_wl_width.ravel()
        y_data_1d = spec_y_data.ravel()
        opt_weight_1d = weight_data.ravel()

        n_out_y = y2 - y1 + 4
        y_min = y1 - 2.
        drizzled_flux = numpy.zeros((out_wl_count, n_out_y))
        drizzled_var = numpy.zeros((out_wl_count, n_out_y))
        drizzled_npix = numpy.zeros((out_wl_count, n_out_y))

        for px in range(wl_width_1d.shape[0]):

            # find first and last pixel in output array to receive some of the flux of this input pixel
            first_px = (wl_from_1d[px] - wl0) / dwl
            last_px = (wl_to_1d[px] - wl0) / dwl

            opt_weight = 1.0
            if (optimal_weight is not None):
                opt_weight = opt_weight_1d[px]
            if (opt_weight <= 0):
                continue

            #    opt_weight = optimal_weight.get_weight(wl=wl_data_1d[px], y=y_data_1d[px])
            #    #print "optimal weight @", wl_data_1d[px], y_data_1d[px], " == ", opt_weight

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

                iy = int(math.floor(y_data_1d[px] - y_min))

                drizzled_flux[tp,iy] += fraction * img_data_per_wl_1d[px]
                drizzled_var[tp,iy] += fraction * var_data_per_wl_1d[px]
                drizzled_npix[tp,iy] += fraction

        _x,_y = numpy.indices((drizzled_flux.shape), dtype=numpy.float)
        _y += y_min
        _x = _x * dwl + wl0


        #
        # Now in a final step, apply optimal weighting
        #
        if (optimal_weight is not None):
            opt_weights_drizzled = optimal_weight.get_weight(
                    wl=_x, y=_y)
        else:
            opt_weights_drizzled = numpy.ones_like(drizzled_flux)

        numpy.savetxt("drizzled_spec.2d.%d-%d" % (y1,y2),
                      numpy.array([_x.ravel(),
                                   _y.ravel(),
                                   drizzled_flux.ravel(),
                                   drizzled_var.ravel(),
                                   opt_weights_drizzled.ravel(),
                                   drizzled_npix.ravel(),
                                   ]).T
                      )

        spec_1d_sum = numpy.nansum(drizzled_flux, axis=1)
        numpy.savetxt("drizzled_spec.simple.%d-%d" % (y1,y2), spec_1d_sum)

        _spec_1d_weighted = \
            numpy.nansum((drizzled_flux * opt_weights_drizzled),
                         axis=1)
        _weight_times_npix = (opt_weights_drizzled * drizzled_npix)
        _spec_1d_weights = numpy.nansum(_weight_times_npix, axis=1) / \
                           numpy.nansum(drizzled_npix, axis=1)
        # spec_1d_optimal = numpy.nansum(
        #     (drizzled_flux * opt_weights_drizzled), axis=1) / numpy.sum(
        #     opt_weights_drizzled, axis=1)
        spec_1d_optimal = _spec_1d_weighted / _spec_1d_weights
        print spec_1d_optimal.shape
        numpy.savetxt("drizzled_spec.1d.%d-%d" % (y1,y2), spec_1d_optimal)



        #
        # compute the scaling factor from plain integration to optimally
        # weighted average extraction.
        #
        flux_scaling = numpy.nanmedian(spec_1d_sum / spec_1d_optimal)
        logger.info("Scaling factor from optimally weighted average to "
                    "simple sum: %f" % (flux_scaling))

        print type(flux_scaling)
        spec_1d_final = spec_1d_optimal * flux_scaling
        numpy.savetxt("drizzled_spec.final.%d-%d" % (y1,y2), spec_1d_final)

        #
        # Repeat the same arithmetic for the variance data
        #
        # TODO: CHECK THAT THE SCALING OF THE VARIANCE DATA IS VALID !!!
        #
        var_1d_optimal = numpy.nansum(
            (drizzled_var * opt_weights_drizzled), axis=1) / numpy.sum(
            opt_weights_drizzled, axis=1)
        var_1d_sum = numpy.nansum(drizzled_var, axis=1)
        var_scaling = numpy.nanmedian(var_1d_sum / var_1d_optimal)
        var_1d_final = var_1d_optimal * var_scaling

        # x = numpy.empty((out_flux.shape[0],5))
        # x[:,0] = numpy.arange(out_flux.shape[0], dtype=numpy.float)*dwl+wl0
        # x[:,1] = out_flux[:]
        # x[:,2] = out_weight[:]
        # x[:,3] = out_var[:]
        # x[:,4] = out_weight2[:]
        # numpy.savetxt("shit", x)
        #
        # #
        # # done with this 1-d extracted spectrum
        # #
        # numpy.savetxt("opt_weight", out_weight)
        # if (optimal_weight is None):
        #     out_weight = numpy.ones_like(out_flux)

        spectra_1d[:, cur_yrange, 0] = spec_1d_final[:]
        spectra_1d[:, cur_yrange, 1] = spec_1d_optimal[:]
        spectra_1d[:, cur_yrange, 2] = spec_1d_sum[:]

        variance_1d[:, cur_yrange, 0] = var_1d_optimal[:]
        variance_1d[:, cur_yrange, 1] = var_1d_sum[:]
        variance_1d[:, cur_yrange, 2] = var_1d_final[:]
        logger.debug("Done with 1-d extraction")

    #
    # Finally, merge wavelength data and flux and write output to file
    #
    output_format = "ascii"
    out_fn = "opt_extract.dat"
    if (output_format.lower() == "fits"):
        logger.info("Writing FITS output to %s" % (out_fn))
        # create the output multi-extension FITS
        spec_3d = numpy.empty((2, spectra_1d.shape[1], spectra_1d.shape[0]))
        spec_3d[0, :, :] = spectra_1d.T[:, :]
        spec_3d[1, :, :] = variance_1d.T[:, :]

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
            for i, yr in enumerate(y_ranges):
                keyname = "YR_%03d" % (i + 1)
                value = "%04d:%04d" % (yr[0], yr[1])
                hdulist[ext].header[keyname] = (value, "y-range for aperture %d" % (i + 1))
        hdulist.writeto(out_fn, clobber=True)
    else:
        logger.info("Writing output as ASCII to %s / %s.var" % (out_fn, out_fn))

        with open(out_fn, "w") as of:
            for aper, yr in enumerate(y_ranges):
                print >>of, "# APERTURE: ", yr
                numpy.savetxt(of, numpy.append(out_wl.reshape((-1, 1)),
                                               spectra_1d[:,aper,:],
                                               axis=1
                                               )
                              )
                print >>of, "\n"*5

        with open(out_fn+".var", "w") as of:
            for aper, yr in enumerate(y_ranges):
                print >>of, "# APERTURE: ", yr
                numpy.savetxt(of, numpy.append(out_wl.reshape((-1, 1)),
                                               variance_1d[:,aper,:],
                                               axis=1
                                               )
                              )
                print >>of, "\n"*5
        # numpy.savetxt(out_fn + ".var",
        #               numpy.append(out_wl.reshape((-1, 1)),
        #                            variance_1d, axis=1))

    logger.debug("All done!")
    # numpy.savetxt(out_fn+".perwl",
    #               numpy.append(wl_data_1d.reshape((-1,1)),
    #                            img_data_per_wl_1d.reshape((-1,1)), axis=1))
    # numpy.savetxt(out_fn + ".raw",
    #           numpy.append(wl_data_1d.reshape((-1, 1)),
    #                        img_data.reshape((-1, 1)), axis=1))

    return spectra_1d, variance_1d

def dummy():
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

        pixels_y = range(int(math.floor(first_px)), int(math.ceil(last_px)))  # , dtype=numpy.int)
        # print wl_data_1d[px], wl_width_1d[px], first_px, last_px, pixels
        for i, tp in enumerate(pixels_y):

            if (i == 0 and i == len(pixels_y) - 1):
                # all flux in a single output pixel
                fraction = last_px - first_px
            elif (i == 0):
                # first pixel, with more to come
                fraction = (tp + 1.) - first_px
            elif (i == len(pixels_y) - 1):
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

if __name__ == "__main__":

    # fn = sys.argv[1]
    # print fn

    # try to look up indices
    wl_start = 5000
    wl_step = 1
    y_start = -50
    y_step = 1
    data = numpy.zeros((500, 100))

    opt = OptimalWeight(
        wl_min=wl_start, wl_step=wl_step,
        y_min=y_start, y_step=y_step,
        data=data)

    print "\n\n\nTrying single value"
    w = opt.get_weight(wl=5005, y=0)
    print w

    print "\n\n\nTrying 1-d array value"
    w = opt.get_weight(wl=[5005, 5020, 5050], y=[0, -5, +10])
    print w


    print "\n\n\nTrying 2-d array"
    wl = [[5005, 5007, 4999], [5002, 5009, 11000]]
    y = [[0, 5, -10], [-76, 100, 0]]
    w = opt.get_weight(wl=wl, y=y)
    print w


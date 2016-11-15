#!/usr/bin/env python


import os
import sys
import astropy.io.fits as fits
import numpy
import math
import itertools

import find_sources
import tracespec

class OptimalWeight(object):

    def __init__(self, wl_min, wl_step, y_min, y_step, data):
        self.wl_min = wl_min
        self.wl_step = wl_step
        self.y_min = y_min
        self.y_step = y_step
        self.data = data

    def get_weight(self, wl, y):

        # find the closest
        return 1.


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


import scipy
import scipy.interpolate

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

    y,x = numpy.indices(out_drizzle.shape)
    combined = numpy.empty((x.shape[0]*x.shape[1], 3))
    combined[:,0] = x.flatten()
    combined[:,1] = y.flatten()
    combined[:,2] = out_drizzle.flatten()
    numpy.savetxt("drizzled", combined)
    numpy.savetxt("drizzled2", out_drizzle)

    #
    # Now integrate the drizzle spectrum along the slit to compute the relative contribution of each pixel.
    #
    spec1d = numpy.sum(out_drizzle, axis=0)
    print out_drizzle.shape, spec1d.shape
    numpy.savetxt("spec1d", spec1d)

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

    #
    # Now we have a full 2-d distribution of extraction weights as fct. of dy and wavelength
    #
    print "computing final 2-d interpolator"
    # weight_interpol = scipy.interpolate.interp2d(
    #     x=combined[:,0],
    #     y=combined[:,1],
    #     z=combined[:,2],
    #     kind='linear',
    #     copy=True,
    #     bounds_error=False,
    #     fill_value=0.,
    # )
    weight_interpol = scipy.interpolate.RectBivariateSpline(
        x=y_start,
        y=wl_start,
        z=drizzled_weight,
        kx=1, ky=1,
    )
    print weight_interpol
    return None

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

    fn = sys.argv[1]
    print fn





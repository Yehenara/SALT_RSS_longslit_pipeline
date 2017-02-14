#!/bin/env python




from astropy.io import fits
import os
import sys
import numpy
import math
import logging
import bottleneck

from pysalt import mp_logging

def map_distortions(wl_2d, diff_2d, img_2d, y, x_list, badrows=None,
                    debug=False, dwl=3):

    distortions = [None] * len(x_list)
    distortions_binned = [None] * len(x_list)
    distortions_median = [None] * len(x_list)
    distortions_std = [None] * len(x_list)

    logger = logging.getLogger("MapDistortion")

    img_2d = numpy.array(img_2d)

    #
    # Create a bias profile
    #
    masked = numpy.array(img_2d)
    masked[~numpy.isfinite(masked)] = 1e9
    bias_level = numpy.percentile(masked, 3)
    logger.debug("Correcting input for global bias level: %f" % (bias_level))

    masked -= bias_level
    img_2d -= bias_level


    for i_dist, x in enumerate(x_list):

        ix = int(x-1)
        iy = int(y-1)

        wl = wl_2d[iy,ix]
        # print x,y,"-->",wl
        logger.debug("Mapping distortions around x=%4d/y=%4d (lambda=%7.2f A)"
                  % (x,y,wl))

        _y, _x = numpy.indices(wl_2d.shape)

        # dwl = 3.

        datastore = numpy.empty((wl_2d.shape[0], 14))
        datastore[:,:] = numpy.NaN

        for cur_y in range(wl_2d.shape[0]):
            in_range = (wl_2d[cur_y, :] > (wl-dwl)) & (wl_2d[cur_y, :] < (wl+dwl))

            if (badrows is not None and badrows[cur_y]):
                continue

            if (numpy.sum(in_range) <= 0):
                continue

            left_x = numpy.min(_x[cur_y, in_range])
            right_x = numpy.max(_x[cur_y, in_range])

            img_cutout = img_2d[cur_y, left_x:right_x+1]
            wl_cutout = wl_2d[cur_y, left_x:right_x+1]

            if (numpy.sum(numpy.isnan(img_cutout)) > 0):
                continue

            # now find the extrema pixels

            peak = numpy.argmax(img_cutout)
            peak_wl = wl_cutout[peak]
            peak_flux = img_cutout[peak]

            if (diff_2d is not None):
                diff_cutout = diff_2d[cur_y, left_x:right_x+1]
                black = numpy.argmin(diff_cutout)
                white = numpy.argmax(diff_cutout)
            else:
                black, white = numpy.NaN, numpy.NaN


            # if ((cur_y % 100) == 0):
            #     print cur_y, left_x, right_x, numpy.sum(in_range), img_cutout.shape, black, white, peak

            half_peak = 0.5* peak_flux
            n_pixels = numpy.sum(half_peak)
            min_intensity = numpy.min(img_cutout)
            total_flux = numpy.sum(img_cutout[img_cutout>half_peak])

            datastore[cur_y,:] = [cur_y, left_x, right_x, img_cutout.shape[0], black, white, peak, wl, x, peak_wl, peak_flux, total_flux, n_pixels, min_intensity]
            # columns are:
            # Col 01 / Idx 00: Y
            # col 02 /     01: left edge
            #     03 /     02: right edge
            #     04 /     03: #pixels
            #     05 /     04: lowest pixel in diff. image
            #     06 /     05: highest pixel in diff. image
            #     07 /     06: peak position in index of sub-pixel
            #     08 /     07: wl of line
            #     09 /     08: x pos of peak
            #     10 /     09: peak wl
            #     11 /     10: peak flux
            #     12 /     11: integrated flux
            #     13 /     12: number of pixels with f>0.5*f_peak
            #     14 /     13: min_intensity

        #
        # Now that we have the full profile, compute mean profile every N rows
        #
        n_rows = 25
        n_max = int(math.floor(datastore.shape[0]/n_rows))*n_rows
        data_trunc = datastore[:n_max, :]
        binned = numpy.nanmean(data_trunc.reshape((n_max/n_rows, n_rows, data_trunc.shape[1])), axis=1)
        # print binned.shape, data_trunc.shape, datastore.shape

        binned_std = bottleneck.nanstd(data_trunc.reshape((n_max/n_rows, n_rows, data_trunc.shape[1])), axis=1)
        binned_median = bottleneck.nanmedian(data_trunc.reshape((n_max/n_rows, n_rows, data_trunc.shape[1])), axis=1)
        if (debug):
            numpy.savetxt("distortion_%d.log" % (x), datastore)
            numpy.savetxt("distortion_%d.bin" % (x), binned)
            numpy.savetxt("distortion_%d.binstd" % (x), binned_std)
            numpy.savetxt("distortion_%d.binmed" % (x), binned_median)

        distortions[i_dist] = datastore
        distortions_binned[i_dist] = binned
        distortions_median[i_dist] = binned_median
        distortions_std[i_dist] = binned_std

    return distortions, distortions_binned, bias_level, distortions_median, distortions_std



if __name__ == "__main__":


    logger = mp_logging.setup_logging()


    fn = sys.argv[1]

    hdulist = fits.open(fn)

    wl_2d = hdulist['WAVELENGTH'].data
    diff_2d = hdulist['SKYSUB.OPT'].data
    img_2d = hdulist['SCI'].data

    y = int(sys.argv[2])

    x_list = [int(f) for f in sys.argv[3:]]

    map_distortions(wl_2d, diff_2d, img_2d, y, x_list)

    mp_logging.shutdown_logging(logger)


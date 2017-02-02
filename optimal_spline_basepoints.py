#!/usr/bin/env python

import os, sys, numpy, time
import scipy, scipy.interpolate, scipy.spatial, scipy.ndimage

from astropy.io import fits

import pysalt.mp_logging
import logging
import bottleneck

import wlcal
import skyline_intensity
import logging
import find_edges_of_skylines
import fastedge
import skytrace
import wlmodel
import localnoise
import plot_high_res_sky_spec
import math
import quickwlmodel

lots_of_debug = True

import spline_pickle_test

use_fast_edges = True

lots_of_debug = True#False

def satisfy_schoenberg_whitney(data, basepoints, k=3):

    logger = logging.getLogger("SchoenbergWhitney")

    logger.debug("Starting with %d basepoints" % (basepoints.shape[0]))

    delete = numpy.isnan(basepoints)
    count,bins = numpy.histogram(
        a=data,
        bins=basepoints,
    )
    delete[count == 0] = True
    # logger.debug("done with histogram method, continuing old-fashioned way")

    # for idx in range(basepoints.shape[0]-1):
    #     # count how many data points are between this and the next basepoint
    #     in_range = (data > basepoints[idx]) & (data < basepoints[idx+1])
    #     count = numpy.sum(in_range)
    #     if (count <= k):
    #         # delete this basepoint
    #         delete[idx] = True
    #         logger.debug("BP % 5d: Deleting basepoint @ %.5f (idx, %d data points, < %d)" % (
    #             idx, basepoints[idx], count, k))

    n_delete = numpy.sum(delete)
    logger.debug("Deleting %d basepoints, left with %d" % (
        n_delete, basepoints.shape[0]-n_delete))

    #logger.debug("%s %s" % (str(delete.shape), str(delete2.shape)))
    #logger.debug("old vs new: %s" % ((delete[:-1] == delete2)))

    return basepoints[~delete]

    
def find_source_mask(img_data):

    #
    # Flatten image in wavelength direction
    #
    flat = bottleneck.nanmedian(img_data.astype(numpy.float32), axis=1)
    print img_data.shape, flat.shape

    if (lots_of_debug):
        numpy.savetxt("obj_mask.flat", flat)

    median_level = numpy.median(flat)
    print median_level


    # do running median filter
    med_filt = scipy.ndimage.filters.median_filter(flat.reshape((-1,1)), size=49, mode='mirror')[:,0]
    if (lots_of_debug):
        numpy.savetxt("obj_mask.medfilt", med_filt)

    excess = flat - med_filt
    good = numpy.isfinite(excess, dtype=numpy.bool)
    print good
    print numpy.sum(good)

    combined = numpy.append(numpy.arange(excess.shape[0]).reshape((-1,1)),
                            excess.reshape((-1,1)), axis=1)
    # compute noise

    for i in range(3):
        _med = numpy.median(excess[good])
        _std = numpy.std(excess[good])
        print _med, _std
        good = (excess > _med-3*_std) & (excess < _med+3*_std)
        print numpy.sum(good)
        if (lots_of_debug):
            numpy.savetxt("obj_mask.filter%d" % (i+1), combined[good])
        
    source = ~good
    print source

    source_mask = scipy.ndimage.filters.convolve(
        input=source, 
        weights=numpy.ones((11)), 
        output=None, 
        mode='reflect', cval=0.0)
    print source_mask

    if (lots_of_debug):
        numpy.savetxt("obj_mask.src", combined[source_mask])

    return source_mask

    pass


def find_center_row(data):

    # Create interpolator for the median profile
    interp = scipy.interpolate.interp1d(
        x=data[:,0], y=data[:,1], kind='linear', 
        bounds_error=False, fill_value=numpy.NaN)

    #
    # Optimization routine
    #
    def fold_profile(p, interp, maxy, count):

        dx = numpy.arange(maxy, dtype=numpy.float) 
        x_left = p[0] - dx
        x_right = p[0] + dx

        profile_left = interp(x_left)
        profile_right = interp(x_right)

        diff = profile_left - profile_right

        count[0] += 1
        # print "iteration %d --> %e" % (count[0], p[0])
        # with open("opt_%d.del" % (count[0]), "w") as f:
        #     numpy.savetxt(f, profile_left)
        #     print >>f, "\n"*5,
        #     numpy.savetxt(f, profile_right)
        #     print >>f, "\n"*5,
        #     numpy.savetxt(f, diff)

        return diff[numpy.isfinite(diff)]

    #
    # Get rid of all points that are too noisy
    #
    w=5
    noise = numpy.array([bottleneck.nanvar(data[i-w:i+w,1]) for i in range(w,data.shape[0]-w+1)])
    # numpy.savetxt("median_noise", noise)
    noise[:w] = numpy.NaN
    noise[-w:] = numpy.NaN

    for iteration in range(3):
        valid = numpy.isfinite(noise)
        _perc = numpy.percentile(noise[valid], [16,50,84])
        _med = _perc[1]
        _sigma = 0.5*(_perc[2]-_perc[0])
        outlier = (noise > _med+3*_sigma) | (noise < _med - 3*_sigma)
        noise[outlier] = numpy.NaN

    #numpy.savetxt("median_noise2", noise)
    valid = numpy.isfinite(noise)
    data[:,1][~valid] = numpy.NaN

    #numpy.savetxt("median_noise3", data)

    count=[0]
    fit_all = scipy.optimize.leastsq(
        func=fold_profile,
        x0=[data.shape[0]/5.],
        args=(interp, data.shape[0]/2,count),
        full_output=True,
        epsfcn=1e-1,
        )

    #print fit_all[0]

    return fit_all[0][0]

def optimal_sky_subtraction(obj_hdulist,
                            image_data=None,
                            sky_regions=None,
                            slitprofile=None,
                            N_points = 6000,
                            wlmode='arc',
                            compare=False,
                            iterate=False,
                            return_2d = True,
                            skiplength=1,
                            mask_objects=True,
                            add_edges=True,
                            skyline_flat=None,
                            select_region=None,
                            debug_prefix="",
                            obj_wl=None,
                            noise_mode='global',
                            debug=False):

    logger = logging.getLogger("OptSplineKs")
    skiplength = 1

    lots_of_debug = debug


    #wl_map = wlmap_model

    logger.info("Loading all data from FITS")

    import time
    time.sleep(1)
    #print '\n'*5, 'img_data\n\n\n', image_data.shape, "\n", image_data, '\n'*5
    if (image_data is None):
        obj_data = obj_hdulist['SCI.RAW'].data #/ fm.reshape((-1,1))
        #print "obj_data:", obj_data.shape
        logger.debug("Using obj_data from SCI.RAW extension")
    else:
        obj_data = image_data.copy()
        logger.debug("Using obj_data from passed image data")

    #obj_wl   = wlmap_model #wl_map #obj_hdulist['WAVELENGTH'].data

    x_eff, wl_map, medians, p_scale, p_skew, fm = \
        None, None, None, None, None, numpy.ones(obj_hdulist['SCI'].data.shape[0])
    wlmap_model = None
    y_center = 2070 #obj_hdulist['SCI'].data.shape[0]/2.

    if (obj_wl is not None):
        logger.info("Using existing WL model")
        pass

    elif (wlmode == 'arc'):
        obj_wl = obj_hdulist['WAVELENGTH'].data
        logger.info("Using wavelength solution from ARC")

    elif (wlmode == 'sky'):
        #
        # Prepare a new refined wavelength map by using sky-lines
        #
        
        (x_eff, wl_map, medians, p_scale, p_skew, fm) = skytrace.create_wlmap_from_skylines(obj_hdulist)

        obj_wl = wl_map
        logger.info("Using wavelength solution constructed from SKY lines")
        y_center = find_center_row(medians)
        logger.info("Using row %.1f as center line of focal plane" % (y_center))
    elif (wlmode == 'model'):

        wlmap_model = wlmodel.rssmodelwave(
            header=obj_hdulist[0].header, 
            img=obj_hdulist['SCI'].data,
            xbin=4, ybin=4,
            y_center=y_center)

        obj_wl = wlmap_model
        logger.info("Using synthetic model wavelength map")
    else:
        logger.error("Unknown WL mode, using ARC instead!")
        obj_wl = obj_hdulist['WAVELENGTH'].data

    # obj_wl   = wl_map #wl_map #obj_hdulist['WAVELENGTH'].data
    obj_rms  = obj_hdulist['VAR'].data / fm.reshape((-1,1))

    # store the wavelength map we end up using to return it to the main process
    wl_map = obj_wl

    if (lots_of_debug):
        pysalt.clobberfile(debug_prefix+"XXX.fits")
        obj_hdulist.writeto(debug_prefix+"XXX.fits", clobber=True)

    try:
        obj_spatial = obj_hdulist['SPATIAL'].data
    except:
        logger.warning("Could not find spatial map, using plain x/y coordinates instead")
        obj_spatial, _ = numpy.indices(obj_data.shape)

    # now merge all data frames into a single 3-d numpy array
    logger.info("Combining all data required for 2-D sky estimation")
    obj_cube = numpy.empty((obj_data.shape[0], obj_data.shape[1], 4))
    obj_cube[:,:,0] = obj_wl[:,:]
    obj_cube[:,:,1] = (obj_data*1.0)[:,:] 
    obj_cube[:,:,2] = obj_rms[:,:]
    obj_cube[:,:,3] = obj_spatial[:,:]

    good_sky_data = numpy.isfinite(obj_data)
    #print good_sky_data
    #print "\n\n #1: --\n", good_sky_data.shape, obj_cube.shape, "\n\n"
    if (lots_of_debug):
        pysalt.clobberfile(debug_prefix+"data_preflat.fits")
        fits.PrimaryHDU(data=obj_cube[:,:,1]).writeto(debug_prefix+"data_preflat.fits", clobber=True)

    if (skyline_flat is not None):
        # We also received a skyline flatfield for field flattening
        obj_cube[:,:,1] /= skyline_flat.reshape((-1,1))
        logger.info("Applying skyline flatfield to data before sky-subtraction")
        #return 1,2
        pass

    if (lots_of_debug):
        pysalt.clobberfile(debug_prefix+"data_postflat.fits")
        fits.PrimaryHDU(data=obj_cube[:,:,1]).writeto(debug_prefix+"data_postflat.fits", clobber=True)

    
    # mask_objects = False
    if (not mask_objects and select_region is None):
        obj_bpm  = numpy.array(obj_hdulist['BPM'].data).flatten()
        good_sky_data &= (obj_hdulist['BPM'].data == 0)
        logger.info("Using full-frame (no masking) for sky estimation")

    else:
        
        use4sky = numpy.ones((obj_cube.shape[0]), dtype=numpy.bool)
        # by default, use entire frame for sky
 
        if (mask_objects):
            logger.debug("mask_object was selected, so looking for source_mask next")
            source_mask = find_source_mask(obj_data)
            use4sky = use4sky & (~source_mask)
            # trim down sky by regions not contaminated with (strong) sources

        if (select_region is not None):
            logger.debug("limiting sky to selected regions")
            sky = numpy.zeros((obj_cube.shape[0]), dtype=numpy.bool)
            for y12 in select_region:
                #print "@@@@@@@@@@",y12, numpy.sum(use4sky), use4sky.shape
                logger.debug("Selecting sky-region: y =%d--%d (total: %d)" % (
                    y12[0], y12[1], numpy.sum(use4sky)))
                sky[y12[0]:y12[1]] = True
            use4sky = use4sky & sky
            # also only select regions explicitely chosen as sky

        #print "selecting:", use4sky.shape, numpy.sum(use4sky)
        logger.info("Computing sky-spectrum from total of %d pixel-lines after masking" % (numpy.sum(use4sky)))

        #
        # mark all excluded regions as such and exclude them from the sky
        # dataset
        #
        good_sky_data[~use4sky] = False
        # obj_cube = obj_cube[use4sky]

        if (lots_of_debug):
            _x = numpy.array(obj_data)
            _x[source_mask] = numpy.NaN
            pysalt.clobberfile(debug_prefix+"obj_mask.fits")
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(data=obj_data),
                fits.ImageHDU(data=_x)]).writeto(debug_prefix+"obj_mask.fits")

        obj_bpm  = numpy.array(obj_hdulist['BPM'].data)[use4sky].flatten()
        #print obj_bpm.shape, obj_cube.shape
    #print "\n\n #2: --\n", good_sky_data.shape, obj_cube.shape, "\n\n"

    # convert dataset from 3-d to 2-d
    # obj_cube = obj_cube.reshape((-1, obj_cube.shape[2]))

    # Now exclude all pixels marked as bad
    #valid_pixels = (obj_bpm == 0) & numpy.isfinite(obj_cube[:, 1])
    #obj_cube = obj_cube[valid_pixels]

    if (lots_of_debug):
        fits.PrimaryHDU(data=good_sky_data.astype(numpy.int)).writeto(
            "goodskydata.0.fits", clobber=True)

    n_good_pixels = numpy.sum(good_sky_data)

    logger.info("%d pixels (of %d, %.f) left after eliminating bad "
                "pixels!" % (
        n_good_pixels, good_sky_data.size,
        100.*n_good_pixels/good_sky_data.size))


    #
    # Now also exclude all points that are marked as non-sky regions 
    # (e.g. including source regions)
    #
    if (not sky_regions is None and
        type(sky_regions) == numpy.ndarray):

        print sky_regions
        logger.info("Selecting sky-pixels from user-defined regions")
        is_sky = numpy.zeros((obj_cube.shape[0]), dtype=numpy.bool)
        for idx, sky_region in enumerate(sky_regions):
            logger.debug("Good region: %d ... %d" % (sky_region[0], sky_region[1]))
            in_region = (obj_cube[:,3] > sky_region[0]) & \
                        (obj_cube[:,3] < sky_region[1]) & \
                        (numpy.isfinite(obj_cube[:,1]))
            is_sky[in_region] = True

        # obj_cube = obj_cube[is_sky]
    else:
        logger.info("No user-selected sky-regions, using full available frame")

    #print "\n\n #3: --\n", good_sky_data.shape, obj_cube.shape, "\n\n"

    allskies = obj_cube #[::skiplength]
    if (lots_of_debug):
        # numpy.savetxt(debug_prefix+"xxx1", allskies)
        # print obj_cube.shape
        # print good_sky_data.shape

        _x = obj_cube[good_sky_data]
        # print _x.shape
        logger.debug("Saving debug output")
        numpy.savetxt(debug_prefix+"xxx1",
                      obj_cube[good_sky_data].reshape((-1, obj_cube.shape[2])))




    # _x = fits.ImageHDU(data=obj_hdulist['SCI.RAW'].data, 
    #                      header=obj_hdulist['SCI.RAW'].header)
    # _x.name = "STEP1"
    # obj_hdulist.append(_x)



    # #
    # # Load and prepare data
    # #
    # allskies = numpy.loadtxt(allskies_filename)

    # just to be on the safe side, sort allskies by wavelength
    logger.debug("Sorting input data by wavelength")
    allskies = obj_cube[good_sky_data].reshape((-1, obj_cube.shape[2]))
    sky_sort_wl = numpy.argsort(allskies[:,0])
    allskies = allskies[sky_sort_wl]
    if (lots_of_debug): numpy.savetxt(debug_prefix+"xxx2", allskies[::skiplength])


    logger.debug("Working on %7d data points to estimate sky" % (allskies.shape[0]))


    #
    # Compute cumulative distribution
    #
    logger.debug("Computing cumulative distribution")
    allskies_cumulative = numpy.cumsum(allskies[:,1], axis=0)

    # print allskies.shape, allskies_cumulative.shape, wl_sorted.shape

    if (lots_of_debug):
        numpy.savetxt(debug_prefix+"cumulative.asc",
                  numpy.append(allskies[::skiplength][:,0].reshape((-1,1)),
                               allskies_cumulative[::skiplength].reshape((-1,1)),
                               axis=1)
                  )
    logger.debug("Cumulative flux range: %f ... %f" % (
        allskies_cumulative[0], allskies_cumulative[-1]))

    # os._exit(0)

    #############################################################################
    #
    # Now create the basepoints by equally distributing them across the 
    # cumulative distribution. This  naturally puts more basepoints into regions
    # with more signal where more precision is needed
    #
    #############################################################################

    # Create a simple interpolator to make life a bit easier
    interp = scipy.interpolate.interp1d(
        x=allskies_cumulative,
        y=allskies[:,0],
        kind='nearest',
        bounds_error=False,
        fill_value=-9999,
        #assume_sorted=True,
        )

    # now create the raw basepoints in cumulative flux space
    k_cumflux = numpy.linspace(allskies_cumulative[0],
                               allskies_cumulative[-1],
                               N_points+2)[1:-1]

    # and using the interpolator, convert flux space into wavelength
    k_wl = interp(k_cumflux)
    logger.debug("Average basepoint spacing: %f A" % ((k_wl[-1]-k_wl[0])/k_wl.shape[0]))

    # eliminate all negative-wavelength basepoints - 
    # these represent interpolation errors
    k_wl = k_wl[k_wl>0]

    if (lots_of_debug):
        numpy.savetxt(debug_prefix+"opt_basepoints",
                  numpy.append(k_wl.reshape((-1,1)),
                               k_cumflux.reshape((-1,1)),
                               axis=1)
                  )
    logger.debug("Done selecting %d spline base points" % (k_wl.shape[0]))

    #############################################################################
    #
    # Add additional wavelength sampling points along the line-edges if
    # this was requested. 
    #
    #############################################################################
    if (add_edges):
        logger.info("Adding sky-samples for line edges")
        
        
        dl = 3.
        dn = 50

        if (use_fast_edges):
            logger.info("Using fast-edge method")
            edges = fastedge.find_line_edges(allskies, line_sigma=2.75)

            # distribute additional basepoints across 2. (+/- dl) angstroem 
            # for each edge
            all_edge_points = numpy.empty((edges.shape[0], dn))
            for ie, edge in enumerate(edges):
                bp = numpy.linspace(edge-dl, edge+dl, dn)
                all_edge_points[ie,:] = bp[:]

        else:
            pysalt.clobberfile(debug_prefix+"edges.cheat")
            if (not os.path.isfile(debug_prefix+"edges.cheat")):
                edges = find_edges_of_skylines.find_edges_of_skylines(allskies, fn="XXX")
                numpy.savetxt(debug_prefix+"edges.cheat", edges)
            else:
                edges = numpy.loadtxt(debug_prefix+"edges.cheat")

            # distribute additional basepoints across 2. (+/- dl) angstroem 
            # for each edge
            all_edge_points = numpy.empty((edges.shape[0], dn))
            for ie, edge in enumerate(edges[:,0]):
                bp = numpy.linspace(edge-dl, edge+dl, dn)
                all_edge_points[ie,:] = bp[:]
        
        # 
        # Now merge the list of new basepoints with the existing list.
        # sort this list ot make it a suitable input for spline fitting
        #
        if (lots_of_debug):
            numpy.savetxt(debug_prefix+"k_wl.in", k_wl)
        k_wl_new = numpy.append(k_wl, all_edge_points.flatten())
        k_wl = numpy.sort(k_wl_new)
        if (lots_of_debug):
            numpy.savetxt(debug_prefix+"k_wl.out", k_wl)

    #############################################################################
    #
    # Now we have the new optimal set of base points, let's compare it to the 
    # original with the same number of basepoints, sampling the available data
    # with points equi-distant in wavelength space.
    #
    #############################################################################


    wl_min, wl_max = numpy.min(allskies[:,0]), numpy.max(allskies[:,0])
    logger.info("Found Min/Max WL-range: %.3f / %.3f" % (wl_min, wl_max))

    if (compare):
        logger.info("Computing spline using original/simple sampling")
        wl_range = wl_max - wl_min
        k_orig_ = numpy.linspace(wl_min, wl_max, N_points+2)[1:-1]
        k_orig = satisfy_schoenberg_whitney(allskies[:,0], k_orig_, k=3)
        spline_orig = scipy.interpolate.LSQUnivariateSpline(
            x=allskies[:,0], 
            y=allskies[:,1], 
            t=k_orig,
            w=None, # no weights (for now)
            #bbox=None, #[wl_min, wl_max], 
            k=3, # use a cubic spline fit
            )
        if (lots_of_debug):
            numpy.savetxt(debug_prefix+"spline_orig", numpy.append(k_orig.reshape((-1,1)),
                                                  spline_orig(k_orig).reshape((-1,1)),
                                                  axis=1)
                      )

    logger.info("Computing spline using optimized sampling")
    logger.debug("#datapoints: %d, #basepoints: %d" % (
        allskies.shape[0], k_wl.shape[0]))

    k_opt_good = satisfy_schoenberg_whitney(allskies[:,0], k_wl, k=3)

    if (lots_of_debug):
        logger.debug("Saving debug output")
        numpy.savetxt(debug_prefix+"allskies", allskies)
        fits.PrimaryHDU(data=allskies).writeto("allskies.fits", clobber=True)
        numpy.savetxt(debug_prefix+"bp_in", k_wl)
        numpy.savetxt(debug_prefix+"bp_out", k_opt_good)
        logger.debug("doe with debug output")

    logger.info("Computing optimized sky-spectrum spline interpolator (%d data, %d base-points)" % (
        allskies.shape[0], k_opt_good.shape[0]
    ))
    try:
        spline_opt = scipy.interpolate.LSQUnivariateSpline(
            x=allskies[:,0], 
            y=allskies[:,1], 
            t=k_opt_good[::10], #k_wl,
            w=None, # no weights (for now)
            bbox=[wl_min, wl_max], 
            k=3, # use a cubic spline fit
        )
    except ValueError:
        logger.error("ERROR: Unable to compute LSQUnivariateSpline (data: %d, bp=%d/10)" % (
            allskies.shape[0], k_opt_good.shape[0]))
        return None, None

    spec_simple = numpy.append(k_wl.reshape((-1,1)),
                                             spline_opt(k_wl).reshape((-1,1)),
                                             axis=1)
    if (lots_of_debug):
        numpy.savetxt(debug_prefix+"spline_opt", spec_simple)

    #
    #
    # Now we have a pretty good guess on what the entire sky spectrum looks like
    # This means we can use known sky-lines to find and compensate for intensity 
    # variations
    #
    #
    if (not iterate):
        if (return_2d):
            pass
        else:
            # only return a 1-d spectrum, centered on the middle row 
            line = obj_wl.shape[0]/2
            wl = obj_wl[line,:]
            spec = spline_opt(wl)

            return spec


    # _x = fits.ImageHDU(data=obj_hdulist['SCI.RAW'].data, 
    #                      header=obj_hdulist['SCI.RAW'].header)
    # _x.name = "STEP2"
    # obj_hdulist.append(_x)


    logger.info("Computing spline using optimized sampling and outlier rejection")
    good_point = (allskies[:,0] > 0)
    logger.info("Using a total of %d pixels for sky estimation" % (good_point.shape[0]))
    #print good_point.shape
    #print good_point

    avg_sample_width = (numpy.max(k_wl) - numpy.min(k_wl)) / k_wl.shape[0]

    # good_data = allskies[good_point]

    spline_iter = None
    n_iterations = 3
    logger.info("Starting iteratively (%dx) computing best sky-spectrum, "
                "using noise-mode %s" % (n_iterations, noise_mode))

    strong_gradient_basepoints_added = False

    for iteration in range(n_iterations):

        logger.info("Starting sky-spectrum iteration %d of %d" % (
            iteration+1, n_iterations)
        )
        # compute spline
        # k_iter_good = satisfy_schoenberg_whitney(good_data[:,0], k_wl, k=3)
        good_data = obj_cube[good_sky_data]
        # print "***\n"*5,good_data.shape,"\n***"*5

        # we now need to sort the data by wavelength
        logger.debug("Sorting input data for iteration %d" % (iteration+1))
        si = numpy.argsort(good_data[:,0])
        good_data = good_data[si]

        #
        # Search for regions of large scatter - these indicate something is
        # not yet well described by the fit and thus could benefit from
        # another spline basepoint in that region.
        #
        logger.info("Searching for additional spline basepoints")
        noiseblocksize = 250
        # print good_data.shape

        n_noise_blocks = math.ceil(good_data.shape[0] / float(noiseblocksize))
        n_to_add = n_noise_blocks*noiseblocksize-good_data.shape[0]
        n_add_front = int(n_to_add/2)
        n_add_back = n_to_add - n_add_front
        pad_width = ((int(n_add_front),int(n_add_back)),(0,0))
         #print pad_width
        prep4noise = numpy.pad(
            good_data,
            pad_width=pad_width,
            mode='constant',
            constant_values=(numpy.NaN,),
        )
        prep4noise_reshape = numpy.reshape(prep4noise,
            (-1, noiseblocksize, good_data.shape[1]))
        # print prep4noise.shape, prep4noise_reshape.shape

        for i in range(1):

            noise_stats = numpy.nanpercentile(
                prep4noise_reshape, [16,50,84], axis=1
            )
            # print noise_stats.shape

            noise_one_sigma = noise_stats[2, :, :] - noise_stats[0, :, :]
            noise_median = noise_stats[1, :, :]
            # print noise_median.shape, noise_one_sigma.shape

            shape_1d = (noise_one_sigma.shape[0], 1, noise_one_sigma.shape[1])
            _good_max = (noise_median + 3 * noise_one_sigma)
            _good_min = (noise_median - 3 * noise_one_sigma)
            bad_data = (prep4noise_reshape > _good_max.reshape(shape_1d)) |  \
                       (prep4noise_reshape < _good_min.reshape(shape_1d))
            prep4noise_reshape[bad_data] = numpy.NaN

            # print noise_stats.shape
            logger.debug("done computing noise spectrum")
            numpy.savetxt("noisespec_%d.%d" % (iteration+1,i+1),
                          numpy.array([noise_median[:,0],
                                       noise_one_sigma[:,1],
                                       noise_median[:,1],
                                       noise_one_sigma[:,1],
                                       ]).T)

        if (not strong_gradient_basepoints_added):
            data = numpy.array([noise_median[:,0],
                                noise_one_sigma[:,1],
                                noise_median[:,1],
                                noise_one_sigma[:,1],
                                ]).T
            basepoints_to_add = quickwlmodel.find_additional_basepoints(
                data=data,
            )
            logger.info("Adding %d spline basepoints in places of strong "
                        "flux-gradients" % (basepoints_to_add.shape[0]))
            k_wl = numpy.sort(numpy.append(k_wl, basepoints_to_add))
            strong_gradient_basepoints_added = True


        logger.debug("Ensuring Schoenberg/Whitney is satisfied")
        k_iter_good = satisfy_schoenberg_whitney(
            good_data[:,0],
            k_wl, k=3)

        logger.info("Computing spline, measuring noise and rejecting "
                    "outliers ...")

        logger.debug("Computing spline ...")
        try:
            # spline_iter = scipy.interpolate.LSQUnivariateSpline(
            #     x=good_data[:,0], #allskies[:,0],#[good_point],
            #     y=good_data[:,1], #allskies[:,1],#[good_point],
            #     t=k_iter_good, #k_wl,
            #     w=None, # no weights (for now)
            #     bbox=[wl_min, wl_max],
            #     k=3, # use a cubic spline fit
            # )
            spline_iter = scipy.interpolate.LSQUnivariateSpline(
                x=good_data[:,0], #allskies[:,0],# #[good_point],
                y=good_data[:,1], #allskies[:,1],
                #  #[good_point],
                t=k_iter_good, #k_wl,
                w=None, # no weights (for now)
                bbox=[wl_min, wl_max],
                k=3, # use a cubic spline fit
            )
            # spline_pickle_test.write_pickle(debug_prefix+"splinein",
            #     fct=scipy.interpolate.LSQUnivariateSpline,
            #     x=good_data[:,0], #allskies[:,0],#[good_point],
            #     y=good_data[:,1], #allskies[:,1],#[good_point],
            #     t=k_iter_good, #k_wl,
            #     w=None, # no weights (for now)
            #     bbox=[wl_min, wl_max],
            #     k=3,
            #     )

        except ValueError as e:
            # this is most likely 
            # ValueError: Interior knots t must satisfy Schoenberg-Whitney conditions
            print e
            if (iteration > 100):
                break
            else:
                logger.warning("unable to compute LSQ spline, skipping 80% of basepoints")
                spline_iter = scipy.interpolate.LSQUnivariateSpline(
                    x=good_data[:,0], #allskies[:,0],#[good_point], 
                    y=good_data[:,1], #allskies[:,1],#[good_point], 
                    t=k_iter_good[5:-5][::5], #k_wl,
                    w=None, # no weights (for now)
                    bbox=[wl_min, wl_max], 
                    k=3, # use a cubic spline fit
                )
            #print("Critical error!")
            #os._exit(0)

        if (lots_of_debug):
            logger.debug("Saving debug data for this iteration")
            numpy.savetxt(debug_prefix+"spline_opt.iter%d" % (iteration+1),
                      numpy.append(k_wl.reshape((-1,1)),
                                   spline_iter(k_wl).reshape((-1,1)),
                                   axis=1)
                  )

        # compute spline fit for each wavelength data point
        # dflux = good_data[:,1] - spline_iter(good_data[:,0])
        logger.debug("computing residuals for outlier rejection")
        modelflux = spline_iter(obj_cube[:,:,0].flatten()).reshape((
            obj_cube.shape[0],obj_cube.shape[1]))
        dflux = obj_cube[:,:,1] - modelflux

        if (lots_of_debug):
            logger.debug("writing dflux.fits")
            fits.PrimaryHDU(data=dflux).writeto(
                "dflux_%d.fits" % (iteration+1),
                clobber=True
            )
            logger.debug("dflux shape: %s" % (str(dflux.shape)))

        # print dflux

        #
        # Add here: work out the scatter of the distribution of pixels in the 
        #           vicinity of this basepoint. This is what determined outlier 
        #           or not, and NOT the uncertainty in a given pixel
        #

        if (noise_mode == 'local1'):

            local_noise = localnoise.calculate_local_noise(
                data=dflux[good_sky_data],
                basepoints=k_iter_good,
                select=good_data[:,0],
                dumpdebug=True
            )
            var = local_noise[:, 0]
            # print var.shape

            # subset_selector = good_data.shape[0] / (100*k_wl.shape[0])
            # subset = good_data[::subset_selector]
            # logger.debug("Searching for points close to each basepoint")
            #
            # # Create a KD-tree with all data points
            # wl_tree = scipy.spatial.cKDTree(subset[:,0].reshape((-1,1)))
            #
            # # Now search this tree for points near each of the spline base points
            # d, i = wl_tree.query(k_wl.reshape((-1,1)),
            #                      k=100, # use 100 neighbors
            #                      distance_upper_bound=avg_sample_width)
            #
            # # make sure to flag outliers
            # bad = (i >= dflux.shape[0])
            # i[bad] = 0
            #
            # # Now we have all indices of a bunch of nearby datapoints, so we can
            # # extract how far off each of the data points is
            # delta_flux_2d = dflux[i]
            # delta_flux_2d[bad] = numpy.NaN
            # #print "dflux_2d = ", delta_flux_2d.shape
            #
            # # With this we can estimate the scatter around each spline fit basepoint
            # logger.debug("Estimating local scatter")
            # var = bottleneck.nanstd(delta_flux_2d, axis=1)
            #print "variance:", var.shape
            if (lots_of_debug):
                numpy.savetxt(debug_prefix+"fit_variance.iter_%d" % (iteration+1),
                          numpy.append(k_iter_good.reshape((-1,1)),
                                       var.reshape((-1,1)), axis=1))

            #
            # Now interpolate this scatter linearly to the position of each
            # datapoint in the original dataset. That way we can easily decide,
            # for each individual pixel, if that pixel is to be considered an
            # outlier or not.
            #
            # Note: Need to consider ALL pixels here, not just the good ones
            #       selected above
            #
            std_interpol = scipy.interpolate.interp1d(
                x = k_iter_good,
                y = var,
                kind = 'linear',
                fill_value=1e3,
                bounds_error=False,
                #assume_sorted=True
                )
            # var_at_pixel = std_interpol(good_data[:,0])
            var_at_pixel = std_interpol(obj_cube[:,:,0])

            logger.debug("Marking outliers for rejection in next iteration")
            if (lots_of_debug):
                numpy.savetxt(debug_prefix+"pixelvar.%d" % (iteration+1),
                          numpy.append(obj_cube[:,:,0].reshape((-1,1)),
                                       var_at_pixel.reshape((-1,1)), axis=1))

            # Now mark all pixels exceeding the noise threshold as outliers

            not_outlier = numpy.fabs(dflux) < 3*var_at_pixel
            outlier = numpy.fabs(dflux) > 3*var_at_pixel


        elif (noise_mode == 'global'):

            #
            # Assume a global noise level everywhere
            # this ignores larger noise around bright emission lines
            #

            sigma = numpy.percentile(dflux[good_sky_data], [16, 84])
            one_sigma = 0.5 * (sigma[1] - sigma[0])
            logger.debug("Found global 1-sigma noise level of %f" % (
                one_sigma))
            not_outlier = numpy.fabs(dflux) < 3*one_sigma
            outlier = numpy.fabs(dflux) > 3*one_sigma

        else:

            #
            # This is a variation of the global noise model
            #

            sigma = numpy.percentile(dflux[good_sky_data], [16, 84])
            # print sigma
            one_sigma = 0.5 * (sigma[1] - sigma[0])
            median = numpy.median(good_data[:,1])
            simple_gain = one_sigma**2 / median

            if (simple_gain < 1 or simple_gain > 3):
                simple_gain = 1.

            logger.debug("Found median=%f, 1-sigma=%f --> gain=%f" % (
                median, one_sigma, simple_gain))

            modelnoise = numpy.sqrt(modelflux * simple_gain + \
                                    one_sigma * simple_gain**2) / simple_gain

            not_outlier = numpy.fabs(dflux) < 3 * modelnoise
            #outlier = numpy.fabs(dflux) > 3 * modelnoise
            outlier = (dflux > 3 * modelnoise) | (dflux < -5*modelnoise)

        #
        # Exclude all outliers from contributing to the next refinement
        # iteration
        #
        good_sky_data[outlier] = False

        #good_data = good_data[not_outlier]
        if (lots_of_debug):
            df = good_data[:,1] - spline_iter(good_data[:,0])
            if (noise_mode == 'local1'):
                _not_outlier = numpy.fabs(df) < std_interpol(good_data[:,0])
            else:
                _not_outlier = numpy.fabs(df) < 3*one_sigma

            # print df.shape, good_data.shape
            debug_fn = debug_prefix+"good_after.%d" % (iteration+1)
            logger.debug("saving debug output to %s" % (debug_fn))
            numpy.savetxt(debug_fn,
                          numpy.array([good_data[:,0][_not_outlier],
                                       good_data[:,1][_not_outlier],
                                       df[_not_outlier]]).T)
                      #
                      #         not_outlier] )
                      # numpy.append(good_data[:,0].reshape((-1,1)),
                      #              dflux[not_outlier].reshape((-1,1)), axis=1))
            fits.PrimaryHDU(data=good_sky_data.astype(numpy.int)).writeto(
                "good_sky_data_after_%d.fits" % (iteration+1), clobber=True)
            logger.debug("done!")


        if (lots_of_debug):
            logger.debug("Creating sky-spec plot")
            plot_high_res_sky_spec.plot_sky_spectrum(
                wl=obj_wl,
                flux=obj_data,
                good_sky_data=good_sky_data,
                bad_rows=None,
                output_filebase="skyspec.iteration_%d" % (iteration+1),
                sky_spline=spline_iter,
                ext_list=['png'],
                basepoints=k_iter_good,
            )

        # logger.info("Done with iteration %d (%d pixels left)" % (
        # iteration+1, good_data.shape[0]))
        logger.info("Done with iteration %d (%d pixels left)" % (
            iteration+1, numpy.sum(good_sky_data)))

    logger.info("Done with all iterative sky-spline fitting (using local noise to reject outliers)")
    # _x = fits.ImageHDU(data=obj_hdulist['SCI.RAW'].data, 
    #                      header=obj_hdulist['SCI.RAW'].header)
    # _x.name = "STEP3"
    # obj_hdulist.append(_x)

    if (compare):
        #
        # Compute differences between data and spline fit for both cases
        #
        logger.info("computing comparison data")
        fit_orig = spline_orig(allskies[:,0])
        fit_opt = spline_opt(allskies[:,0])

        comp = numpy.zeros((allskies.shape[0], allskies.shape[1]+2))
        comp[:, :allskies.shape[1]] = allskies[:,:]
        comp[:, allskies.shape[1]+0] = fit_orig[:]
        comp[:, allskies.shape[1]+1] = fit_opt[:]
        numpy.savetxt(debug_prefix+"allskies.comp", comp)


    #
    # Now in a final step, compute the 2-D sky spectrum, subtract, and save results
    #
    sky2d = None

    logger.info("Computing full-resolution sky spectrum from sky-spline")
    # compute high-res sky-spectrum
    wl_highres = numpy.linspace(allskies[0,0], allskies[-1,0], 100000)
    sky_highres = spline_iter(wl_highres)
    numpy.savetxt(debug_prefix+"sky_highres", numpy.append(wl_highres.reshape((-1,1)),
                                              sky_highres.reshape((-1,1)), axis=1))

    allskies_synth = spline_iter(allskies[:,0])
    ascombined = numpy.zeros((allskies_synth.shape[0],4))
    ascombined[:,0] = allskies[:,0]
    ascombined[:,1] = allskies[:,1]
    ascombined[:,2] = allskies_synth[:]
    ascombined[:,3] = allskies[:,1] - allskies_synth[:]
    numpy.savetxt(debug_prefix+"skysub_all", ascombined)

    if (not spline_iter == None):
        padded = numpy.empty((obj_wl.shape[0], obj_wl.shape[1]+2))
        padded[:, 1:-1] = obj_wl[:,:]
        padded[:,0] = obj_wl[:,0]
        padded[:,-1] = obj_wl[:,-1]
        from_wl = 0.5*(padded[:, 0:-2] + padded[:, 1:-1])
        to_wl = 0.5*(padded[:, 1:-1] + padded[:, 2:])
        # print "n\nXXXXX",from_wl.shape, to_wl.shape, obj_wl.shape, padded, "\nXXXXX\n"

        # this would be a nice call, but xxx.integral does not support multiple values
        # sky2d_x = spline_iter.integral(from_wl.ravel(), to_wl.ravel()).reshape(from_wl.shape)
        #
        # therefore we need to do the integration for each pixel by hand
        t0 = time.time()
        sky2d = numpy.array([spline_iter.integral(a,b) for a,b in zip(from_wl.ravel(),to_wl.ravel())]).reshape(obj_wl.shape)

        #
        # Important !!!
        #
        # the integral routine somehow does some normalization - therefore scaling only works
        # if we divide the final map by the dispersion, i.e. the width of each pixel integral
        # (see prototype code in fiddle_skyspline.py)
        #
        wl_width = to_wl - from_wl
        sky2d /= wl_width

        t1 = time.time()
        logger.debug("Integration took %.3f seconds" % (t1-t0))

        # print "integration took %f seconds" % (t1-t0)
        if (lots_of_debug):
            fits.PrimaryHDU(data=sky2d).writeto(debug_prefix+"IntegSky.fits", clobber=True)
            logger.debug("2d-sky spectrum written to IntegSky.fits")

            fits.PrimaryHDU(data=from_wl).writeto(debug_prefix+"IntegSky_from.fits", clobber=True)
            fits.PrimaryHDU(data=to_wl).writeto(debug_prefix+"IntegSky_to.fits", clobber=True)
            fits.PrimaryHDU(data=(to_wl-from_wl)).writeto(debug_prefix+"IntegSky_delta.fits", clobber=True)

        # t0 = time.time()
        # sky2d = spline_iter(obj_wl.ravel()).reshape(obj_wl.shape)
        # t1 = time.time()
        # print "interpolation took %f seconds" % (t1-t0)

        #if (not type(skyline_flat) == type(None)):
        #    sky2d *= skyline_flat.reshape((-1,1))

        # skysub = obj_data - sky2d
        # ss_hdu = fits.ImageHDU(header=obj_hdulist['SCI.RAW'].header,
        #                          data=skysub)
        # ss_hdu.name = "SKYSUB.OPT"
        # obj_hdulist.append(ss_hdu)

        # ss_hdu2 = fits.ImageHDU(header=obj_hdulist['SCI.RAW'].header,
        #                          data=sky2d)
        # ss_hdu2.name = "SKYSUB.IMG"
        # obj_hdulist.append(ss_hdu2)


        return sky2d, spline_iter, (x_eff, wl_map, medians, p_scale, p_skew,
                                    fm, good_sky_data)

    return None

    



def estimate_slit_intensity_variations(hdulist, spline, sky_2d):

    # We can use a horizontal cut through the 2-D sky as a template of the 
    # sky-spectrum to extract sky emission lines
    row = int(sky_2d.shape[0]/2)
    skyspec = sky_2d[row,:]

    skyline_list = wlcal.find_list_of_lines(skyspec, readnoise=1, avg_width=1)
    print skyline_list
    numpy.savetxt("skyline_list", skyline_list)

    #
    # Select a couple of the strong lines
    #



    #
    # Now trace the arclines and do the subpixel centering so we can compute 
    # the line intensity profiles
    #

    # data = hdulist['RAW'].data
    
    #subpixel_centroid_trace(data, tracedata, width=5, dumpfile=None)



if __name__ == "__main__":


    logger_setup = pysalt.mp_logging.setup_logging()


    obj_fitsfile = sys.argv[1]
    obj_hdulist = fits.open(obj_fitsfile)

    sky_regions = None
    if (len(sys.argv) > 2):
        user_sky = sys.argv[2]
        sky_regions = numpy.array([x.split(":") for x in user_sky.split(",")]).astype(numpy.int)

    # obj_mask = find_source_mask(obj_hdulist['SCI.RAW'].data)

    # simple_spec = optimal_sky_subtraction(obj_hdulist,
    #                                       sky_regions=sky_regions,
    #                                       N_points=1000,
    #                                       iterate=False,
    #                                       skiplength=10,
    #                                       compare=True,
    #                                       mask_objects=True,
    #                                       return_2d=False)
    # numpy.savetxt("simple_spec", simple_spec)
    #
    #
    # skyline_list = wlcal.find_list_of_lines(simple_spec, readnoise=1, avg_width=1)
    # print skyline_list
    #
    # i, ia, im = skyline_intensity.find_skyline_profiles(obj_hdulist, skyline_list)
    #
    # #numpy.savetxt("skyline_list", skyline_list)
    #

    # sky_2d, spline = optimal_sky_subtraction(obj_hdulist,
    #                                          sky_regions=sky_regions,
    #                                          N_points=1000,
    #                                          iterate=False,
    #                                          skiplength=5,
    #                                          mask_objects=True,
    #                                          skyline_flat=ia)

    flattened_img = obj_hdulist['SCI'].data
    wls_2d = obj_hdulist['WAVELENGTH'].data
    sky_2d, spline, extra = optimal_sky_subtraction(
        obj_hdulist,
        sky_regions=None,  # sky_regions,
        N_points=600,
        iterate=False,
        skiplength=5,
        skyline_flat=None, #skyline_flat,  # intensity_profile.reshape((-1,1)),
        # select_region=numpy.array([[900,950]])
        # select_region=numpy.array([[600, 640], [660, 700]]),
        wlmode='model', #options.wlmode,
        debug_prefix="optskysub__",
        image_data=flattened_img,
        obj_wl=wls_2d,
        debug=True, #options.debug,
        noise_mode='global', #options.sky_noise_mode,
    )
    (x_eff, wl_map, medians, p_scale, p_skew, fm, good_sky_data) = extra

    # # Now use the spline interpolator to create a list of strong skylines
    # estimate_slit_intensity_variations(obj_hdulist, spline, sky_2d)

    obj_hdulist.writeto(obj_fitsfile[:-5]+".optimized.fits", clobber=True)

    pysalt.mp_logging.shutdown_logging(logger_setup)

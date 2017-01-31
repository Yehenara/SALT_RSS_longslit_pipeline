#!/usr/bin/env python

import os, sys, numpy, scipy
from astropy.io import fits

import scipy.ndimage
import scipy.stats
import scipy.signal

import pysalt

import logging
import bottleneck


def continuum_slit_profile(hdulist=None, data_ext='SKYSUB.OPT', sky_ext='SKYSUB.IMG', subtract_sky=True,
                           data=None, wl=None, sky=None, var=None):

    logger = logging.getLogger("ContSlitProfile")

    if (hdulist is not None):
        if (data is None):
            data = hdulist[data_ext].data.copy()
        if (wl is None):
            wl = hdulist['WAVELENGTH'].data
        if (sky is None):
            sky = hdulist[sky_ext].data
        if (var is None):
            var = hdulist['VAR'].data.copy()

    if (wl is None or data is None or sky is None or var is None):
        logger.critical("Missing data to compute continuum slit profile")
        return None, None

    fits.PrimaryHDU(data=data).writeto("yyy", clobber=True)
    y = data.shape[0]/2
    #sky1d = sky[y:y+1,:] #.reshape((-1,1))
    sky1d = sky[y,:] #.reshape((-1,1))
    logger.debug("sky1d-shape: %s" % (sky1d.shape))

    wl_1d = wl[y,:]

    mf = scipy.ndimage.filters.median_filter(
        input=sky1d, 
        size=(75), 
        footprint=None, 
        output=None, 
        mode='reflect', 
        cval=0.0, 
        origin=0)
    
    #print mf.shape

    numpy.savetxt("sky", sky1d)
    numpy.savetxt("sky2", mf)

    # pick the intensity of the lowest 10%
    max_intensity = scipy.stats.scoreatpercentile(sky1d, [10,20])
    logger.debug("intensity at 10%%: %s" % (max_intensity))

    #
    # Find emission lines
    #
    line_strength = sky1d - mf
    no_line = numpy.ones(line_strength.shape, dtype=numpy.bool)
    for i in range(3):
        med = numpy.median(line_strength[no_line])
        std = numpy.std(line_strength[no_line])

        no_line = (line_strength < med+2*std) & \
                  (line_strength > med-2*std)
        #print med, std, numpy.sum(no_line)

    #
    # Select regions that do not a sky-line within N pixels
    #
    logger.info("Looking for spectral regions without sky-lines")
    #print line_strength.shape
    N = 15
    buffered = numpy.zeros(line_strength.shape[0]+2*N)
    buffered[N:-N][~no_line] = 1.0
    numpy.savetxt("lines", buffered)

    W = 15
    line_blocker = numpy.array(
        [numpy.sum(buffered[x+N-W:x+N+W]) for x in range(line_strength.shape[0])]
    )
    numpy.savetxt("lineblocker", line_blocker)

    line_contaminated = line_blocker >= 1
    line_strength[line_contaminated] = numpy.NaN
    numpy.savetxt("contsky", line_strength)
    
    #
    # Now we have a very good estimate where skylines contaminate the spectrum
    # We can now isolate regions that are clean to find sources
    #

    #
    # Look for large chunks of spectrum without skylines
    # 
    spec_blocks = []
    blockstart = 0
    in_block = False
    for x in range(1, line_contaminated.shape[0]):
        if (not line_contaminated[x]):
            # this is a good part of spectrum
            if (in_block):
                # we already started this block
                continue
            else:
                in_block = True
                blockstart = x
        else:
            # this is a region close to a skyline
            if (in_block):
                # so far we were in a good strech which is now over
                spec_blocks.append([blockstart, x-1])
                in_block = False
            else:
                continue

    spec_blocks = numpy.array(spec_blocks)
    logger.debug("spec-blocks:\n%s" % (spec_blocks))

    #
    # Now pick a block close to the center of the chip where the spectral
    # trace shows little curvature
    #
    good_blocks = (spec_blocks[:,0] > 0.35*sky1d.shape[0]) & \
                  (spec_blocks[:,1] > 0.65*sky1d.shape[0])
    central_blocks = spec_blocks[good_blocks]
    #print central_blocks

    # Out of these, find the largest one
    block_size = central_blocks[:,1] - central_blocks[:,0]
    #print block_size

    largest_block = numpy.argmax(block_size)

    use_block = central_blocks[largest_block]
    logger.info("Using spectral block for source finding: %s" % (use_block))
        
    wl_min = wl_1d[use_block[0]]
    wl_max = wl_1d[use_block[1]]
    logger.info("wavelength range: %f -- %f" % (wl_min, wl_max))

    out_of_range = (wl < wl_min) | (wl > wl_max)
    data[out_of_range] = numpy.NaN

    var[out_of_range] = numpy.NaN

    fits.PrimaryHDU(data=data).writeto("xxx", clobber=True)

    #intensity_profile = bottleneck.nansum(data, axis=1)
    intensity_profile = numpy.nansum(data, axis=1)
    intensity_error = numpy.nansum(var, axis=1)

    #print data.shape, intensity_profile.shape

    numpy.savetxt("prof", intensity_profile)
    numpy.savetxt("prof.var", intensity_error)

    #
    # Apply wide median filter to subtract continuum slope (if any)
    #

    return intensity_profile, intensity_error


def identify_sources(profile, profile_var=None,
                     min_peak_s2n=4,
                     psf_size=3,
                     pre_median_width=3):

    logger = logging.getLogger("IdentifySources")

    #
    # Smooth profile to get rid of noise spikes
    #
    gauss_width = psf_size
    logger.info("Applying %.1f pixel gauss filter in spectral dir" % (gauss_width))

    #
    # First, apply a small-scale median filter to get rid of single-row
    # spikes due to cosmics
    #
    pre_med = scipy.ndimage.filters.median_filter(
        input=profile,
        size=(pre_median_width),
        footprint=None,
        output=None,
        mode='reflect',
        cval=0.0,
        origin=0)
    numpy.savetxt("prof.premed", pre_med)

    smoothed = scipy.ndimage.filters.gaussian_filter(pre_med.reshape((-1,1)),
                                                     (gauss_width,0),
                                                        mode='constant', cval=0,
    )

    #
    # Approximate a continuum by applying a wider median filter
    #
    cont = scipy.ndimage.filters.median_filter(
        input=pre_med,
        size=(175),
        footprint=None, 
        output=None, 
        mode='reflect', 
        cval=0.0, 
        origin=0)
    
    numpy.savetxt("prof.gauss1", smoothed)
    numpy.savetxt("prof.cont", cont)

    signal = pre_med - cont
    smooth_signal = (smoothed[:,0] - cont)

    #
    # Get a noise-estimate by iteratively rejecting strong features
    #
    valid = numpy.isfinite(signal)
    for iter in range(3):
        stats = scipy.stats.scoreatpercentile(signal[valid], [16,50,84],
                                              limit=[-1e9,1e9])
        one_sigma = (stats[2] - stats[0]) / 2.
        median = stats[1]
        print stats
        #print iter, stats
        # mask all pixels outside the 3-sigma range as bad
        bad = (signal > (median + 3*one_sigma)) | (signal < (median - 3*one_sigma))
        signal[bad] = numpy.NaN
        valid[bad] = False
        numpy.savetxt("signal.%d" % (iter+1), signal)

    #
    # Now we have the noise level, so we can determine where true (or truish) 
    # sources are
    #
    noise_level = numpy.var(signal[numpy.isfinite(signal)])
    logger.info("Noise level: %f / %f" % (one_sigma, noise_level))
    noise_level = one_sigma


    #
    # Compute slope product to test for slope continuity
    #
    slope_input = smoothed[:,0] # smooth_signal
    padded = numpy.zeros((slope_input.shape[0]+2))
    padded[1:-1] = slope_input
    d1 = padded[1:-1] - padded[0:-2]
    d2 = padded[1:-1] - padded[2:]
    slope_product = d1 * d2
    numpy.savetxt("prof.slopeprod", slope_product)

    s2n = smooth_signal / noise_level
    # select raw peak catalog as all peaks with s/n > 3
    print slope_product.shape, slope_input.shape
    valid_peak = (slope_product > 0) & (s2n > min_peak_s2n) & (d1 > 0)
    peak_positions = numpy.arange(slope_input.shape[0])[valid_peak]
    numpy.savetxt("prof.peaks2",
                  numpy.array([peak_positions,
                               slope_input[valid_peak],
                               s2n[valid_peak]
                               ]).T
    )

    # peaks = scipy.signal.find_peaks_cwt(
    #     vector=(smoothed[:,0]-cont),
    #     widths=numpy.array([5]),
    #     wavelet=None,
    #     max_distances=None,
    #     gap_thresh=None,
    #     min_length=None,
    #     min_snr=2,
    #     noise_perc=10,
    #     )
    # peaks = numpy.array(peaks)
    # logger.debug("Raw peak list:\n%s" % (peaks))
    # numpy.savetxt("raw_peaks", peaks)

    #
    # Now we have a bunch of potential sources
    # make sure they are significant enough
    #
    # peak_intensities = smooth_signal[peaks]
    # #print peak_intensities
    # s2n = peak_intensities / noise_level
    # numpy.savetxt("sources.s2n",
    #               numpy.array([peaks, peak_intensities, s2n]).T)
    # #print s2n
    #
    # above threshold
    # significant = (s2n > 3.)
    threshold = 3 * noise_level
    # #print significant
    #
    # #print type(peaks)

    # logger.debug("final list of peaks: %s" % (peaks[significant]))
    # sources = peaks[significant]

    sources = peak_positions
    
    #
    # Now use the slope product to find the maximum extent of each source
    # 
    # To be considered the edge of a source, the pixel intensity needs to 
    # be BELOW the significance threshold, and the slope-product has to be 
    # POSITIVE to mark an upward-trend following a downward-slope or vice versa
    #
    source_stats = numpy.empty((sources.shape[0],4))
    source_stats[:,0] = sources[:]
    source_stats[:,1] = s2n[valid_peak] #s2n[significant]

    pixel_coord = numpy.arange(smooth_signal.shape[0])
    for si, source in enumerate(sources):
        #print source
        out_of_source = (smooth_signal < threshold) & (slope_product > 0)

        on_left = pixel_coord < source
        on_right = pixel_coord > source

        _leftpixels = pixel_coord[out_of_source & on_left]
        left = numpy.max(_leftpixels) if _leftpixels.size > 0 else 0

        _rightpixels = pixel_coord[out_of_source & on_right]
        right = numpy.min(_rightpixels) if _rightpixels.size > 0 else \
            profile.shape[0]
        #print left, right
        #print

        source_stats[si, 2:4] = [left, right]

    #numpy.savetxt(sys.stdout, source_stats, "%.1f")
    numpy.savetxt("sources", source_stats)

    return source_stats


def create_source_tbhdu(sources):

    return fits.BinTableHDU()


def save_continuum_slit_profile(prof, prof_var):

    #
    # Combine both profiles
    #
    combined = numpy.array([prof, prof_var])

    imghdu = fits.ImageHDU(data=combined, name='SRC_PROFILE')

    return imghdu


def write_source_region_file(img_shape, sources, outfile):

    with open(outfile, "w") as src_reg:
        print >> src_reg, """\
    # Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    physical"""
        for source_id, si in enumerate(sources):
            print >> src_reg, "line(0,%d,%d,%d) # line=0 0 " \
                              "color=red text={Source %d}" % (
                                  si[0], img_shape[1], si[0], source_id + 1)
            print >> src_reg, "line(0,%d,%d,%d) # line=0 0 color=green" % (
                si[2], img_shape[1], si[2])
            print >> src_reg, "line(0,%d,%d,%d) # line=0 0 color=green" % (
                si[3], img_shape[1], si[3])

    return


if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()
    logger = logging.getLogger("MAIN")

    hdulist = fits.open(sys.argv[1])

    # profile = continuum_slit_profile(hdulist)
    # sources = identify_sources(profile)

    prof, prof_var = continuum_slit_profile(hdulist)
    numpy.savetxt("source_profile", prof)
    sources = identify_sources(prof, prof_var)

    print "sources:\n",sources

    write_source_region_file(
        img_shape=hdulist['SCI'].data.shape,
        sources=sources,
        outfile="find_sources.reg",
    )
    pysalt.mp_logging.shutdown_logging(logger_setup)


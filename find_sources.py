#!/usr/bin/env python

import os, sys, numpy, scipy
import astropy.io.fits as pyfits

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

    pyfits.PrimaryHDU(data=data).writeto("xxx", clobber=True)

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


def identify_sources(profile, profile_var=None):

    logger = logging.getLogger("IdentifySources")

    #
    # Smooth profile to get rid of noise spikes
    #
    gauss_width = 1
    logger.info("Applying %.1f pixel gauss filter in spectral dir" % (gauss_width))

    smoothed = scipy.ndimage.filters.gaussian_filter(profile.reshape((-1,1)), (gauss_width,0), 
                                                        mode='constant', cval=0,
    )

    #
    # Approximate a continuum by applying a wider median filter
    #
    cont = scipy.ndimage.filters.median_filter(
        input=profile,
        size=(75), 
        footprint=None, 
        output=None, 
        mode='reflect', 
        cval=0.0, 
        origin=0)
    
    numpy.savetxt("prof.gauss1", smoothed)
    numpy.savetxt("prof.cont", cont)

    signal = profile - cont
    smooth_signal = (smoothed[:,0] - cont)

    #
    # Get a noise-estimate by iteratively rejecting strong features
    #
    for iter in range(3):
        stats = scipy.stats.scoreatpercentile(signal, [16,50,84], limit=[-1e9,1e9])
        one_sigma = (stats[2] - stats[0]) / 2.
        median = stats[1]
        #print iter, stats
        # mask all pixels outside the 3-sigma range as bad
        bad = (signal > (median + 3*one_sigma)) | (signal < (median - 3*one_sigma))
        signal[bad] = numpy.NaN
        numpy.savetxt("signal.%d" % (iter+1), signal)

    #
    # Now we have the noise level, so we can determine where true (or truish) 
    # sources are
    #
    noise_level = numpy.var(signal[numpy.isfinite(signal)])
    logger.debug("Noise level: %f / %f" % (one_sigma, noise_level))

    peaks = scipy.signal.find_peaks_cwt(
        vector=(smoothed[:,0]-cont), 
        widths=numpy.array([5]), 
        wavelet=None, 
        max_distances=None, 
        gap_thresh=None, 
        min_length=None, 
        min_snr=2, 
        noise_perc=10,
        )
    peaks = numpy.array(peaks)
    logger.debug("Raw peak list:\n%s" % (peaks))
    
    #
    # Now we have a bunch of potential sources
    # make sure they are significant enough
    #
    peak_intensities = smooth_signal[peaks]
    #print peak_intensities
    s2n = peak_intensities / noise_level
    #print s2n

    # above threshold
    significant = (s2n > 3.)
    threshold = 3 * noise_level
    #print significant

    #print type(peaks)

    logger.debug("final list of peaks: %s" % (peaks[significant]))
    sources = peaks[significant]

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

    
    #
    # Now use the slope product to find the maximum extent of each source
    # 
    # To be considered the edge of a source, the pixel intensity needs to 
    # be BELOW the significance threshold, and the slope-product has to be 
    # POSITIVE to mark an upward-trend following a downward-slope or vice versa
    #
    source_stats = numpy.empty((sources.shape[0],4))
    source_stats[:,0] = sources[:]
    source_stats[:,1] = s2n[significant]

    pixel_coord = numpy.arange(smooth_signal.shape[0])
    for si, source in enumerate(sources):
        #print source
        out_of_source = (smooth_signal < threshold) & (slope_product > 0)

        on_left = pixel_coord < source
        on_right = pixel_coord > source
        left = numpy.max(pixel_coord[out_of_source & on_left])
        right = numpy.min(pixel_coord[out_of_source & on_right])
        #print left, right
        #print

        source_stats[si, 2:4] = [left, right]

    #numpy.savetxt(sys.stdout, source_stats, "%.1f")
    numpy.savetxt("sources", source_stats)

    return source_stats


def create_source_tbhdu(sources):

    return pyfits.BinTableHDU()


def save_continuum_slit_profile(prof, prof_var):

    #
    # Combine both profiles
    #
    combined = numpy.array([prof, prof_var])

    imghdu = pyfits.ImageHDU(data=combined, name='SRC_PROFILE')

    return imghdu


if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()
    logger = logging.getLogger("MAIN")

    hdulist = pyfits.open(sys.argv[1])

    profile = continuum_slit_profile(hdulist)
    sources = identify_sources(profile)


    pysalt.mp_logging.shutdown_logging(logger_setup)


#!/usr/bin/env python


import os
import sys
import astropy.io.fits as fits
import numpy

import find_sources
import tracespec
import optimal_extraction
import pysalt.mp_logging

if __name__ == "__main__":

    logger = pysalt.mp_logging.setup_logging()

    fn = sys.argv[1]

    hdulist = fits.open(fn)

    # compute a source list (includes source position, extent, and intensity)
    prof = find_sources.continuum_slit_profile(hdulist)
    sources = find_sources.identify_sources(prof)

    print sources

    # now pick the brightest of all sources
    i_brightest = numpy.argmax(sources[:,1])
    print i_brightest
    print sources[i_brightest]
    brightest  = sources[i_brightest]

    # Now trace the line
    print "computing spectrum trace"
    spec_data = hdulist['SKYSUB.OPT'].data
    center_x = spec_data.shape[1]/2
    spectrace_data = tracespec.compute_spectrum_trace(
        data=spec_data,
        start_x=center_x,
        start_y=brightest[0],
        xbin=5)

    print "finding trace slopes"
    slopes, trace_offset = tracespec.compute_trace_slopes(spectrace_data)

    print "generating source profile in prep for optimal extraction"
    width = brightest[3]-brightest[2]
    source_profile_2d = optimal_extraction.generate_source_profile(
        data=hdulist['SKYSUB.OPT'].data,
        variance=hdulist['VAR'].data,
        wavelength=hdulist['WAVELENGTH'].data,
        trace_offset=trace_offset,
        position=[center_x, brightest[0]],
        width=width,
    )

    print source_profile_2d

    #
    # Now stack the profile along the spectral direction to compute the integrated profile we need for weighting
    # during the optimal extraction.
    #
    supersample = 2
    optimal_weight = optimal_extraction.integrate_source_profile(
        width=width,
        supersample=supersample,
        profile2d=source_profile_2d,
        wl_resolution=-5,
    )


    optimal_extraction.optimal_extract(
        img_data=hdulist['SKYSUB.OPT'].data,
        wl_data=hdulist['WAVELENGTH'].data,
        variance_data=hdulist['VAR'].data,
        trace_offset=trace_offset,
        optimal_weight=optimal_weight,
        opt_weight_center_y=brightest[0],
        reference_x=center_x,
        reference_y=brightest[0],
        y_ranges=[[-20, 20]],
        dwl=0.5,
    )

    print brightest[0]
    pysalt.mp_logging.shutdown_logging(logger)


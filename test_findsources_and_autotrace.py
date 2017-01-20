#!/usr/bin/env python


import os
import sys
import astropy.io.fits as fits
import numpy

import find_sources
import tracespec
import optimal_extraction
import zero_background
import pysalt.mp_logging

import pickle

if __name__ == "__main__":

    log_setup = pysalt.mp_logging.setup_logging()

    fn = sys.argv[1]

    hdulist = fits.open(fn)

    # compute a source list (includes source position, extent, and intensity)
    prof, prof_var = find_sources.continuum_slit_profile(hdulist)
    numpy.savetxt("source_profile", prof)
    sources = find_sources.identify_sources(prof, prof_var)

    print "="*50,"\n List of sources: \n","="*50
    print sources
    print "\n-----"*5
    with open("sources.reg", "w") as src_reg:
        img = hdulist['SCI'].data
        print >>src_reg, """\
# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
physical"""
        for si in sources:
            print >> src_reg, "line(0,%d,%d,%d) # line=0 0 color=red" % (si[0], img.shape[1], si[0])
            print >> src_reg, "line(0,%d,%d,%d) # line=0 0 color=green" % (si[2], img.shape[1], si[2])
            print >> src_reg, "line(0,%d,%d,%d) # line=0 0 color=green" % (si[3], img.shape[1], si[3])

    fullframe_background = zero_background.find_background_correction(
        hdulist=hdulist,
        sources=sources,
    )
    hdulist['SKYSUB.OPT'].data -= fullframe_background

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
    width = 2*(brightest[3]-brightest[2])
    source_profile_2d = optimal_extraction.generate_source_profile(
        data=hdulist['SKYSUB.OPT'].data,
        variance=hdulist['VAR'].data,
        wavelength=hdulist['WAVELENGTH'].data,
        trace_offset=trace_offset,
        position=[center_x, brightest[0]],
        width=width,
    )

    print "source-profile-2D:\n",source_profile_2d,"\n", source_profile_2d.shape

    #
    # Now stack the profile along the spectral direction to compute the
    # integrated profile we need for weighting during the optimal extraction.
    #

    supersample = 2
    pickle.dump(
        (width, supersample, source_profile_2d), open("optimal_extract.pickle", "wb")
    )
    optimal_weight = optimal_extraction.integrate_source_profile(
        width=width,
        supersample=supersample,
        profile2d=source_profile_2d,
        wl_resolution=-5,
    )
    # optimal weight is an instance of the OptimalWeight class

    y_ranges = [[-25,25]]
    try:
        user_ap = sys.argv[2]
    except:
        user_ap = None
    if (user_ap is not None):
        if (user_ap == "auto"):
            y_ranges = [brightest[2:4] - brightest[0]]
        else:
            y_ranges = []
            for block in user_ap.split(","):
                y12 = [int(d) for d in block.split(":")]
                y_ranges.append(y12[:2])

    print("Extracting sources for these apertures: %s" % (str(y_ranges)))

    d_width = brightest[2:4] - brightest[0]
    optimal_extraction.optimal_extract(
        img_data=hdulist['SKYSUB.OPT'].data,
        wl_data=hdulist['WAVELENGTH'].data,
        variance_data=hdulist['VAR'].data,
        trace_offset=trace_offset,
        optimal_weight=optimal_weight,
        opt_weight_center_y=brightest[0],
        reference_x=center_x,
        reference_y=brightest[0],
        # y_ranges=[[-25, 25]],
        # y_ranges=[[-20, 20]],
        #y_ranges=[d_width],
        y_ranges=y_ranges,
        dwl=0.5,
    )

    print brightest[0]
    pysalt.mp_logging.shutdown_logging(log_setup)


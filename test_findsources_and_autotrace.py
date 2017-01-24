#!/usr/bin/env python


import os
import sys
import astropy.io.fits as fits
import numpy
import logging

import find_sources
import tracespec
import optimal_extraction
import zero_background
import pysalt.mp_logging

import pickle

if __name__ == "__main__":

    log_setup = pysalt.mp_logging.setup_logging()
    logger = logging.getLogger("Trace_&_Extract")

    fn = sys.argv[1]

    out_fn = sys.argv[2]

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
        user_ap = sys.argv[3]
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
    results = optimal_extraction.optimal_extract(
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

    #
    # extract individual data from return data
    #
    spectra_1d = results['spectra']
    variance_1d = results['variance']
    wl0 = results['wl0']
    dwl = results['dwl']
    out_wl = results['wl_base']

    #
    # Finally, merge wavelength data and flux and write output to file
    #
    output_format = ["ascii", "fits"]
    #out_fn = "opt_extract"
    if ("fits" in output_format or True):
        out_fn_fits = out_fn + ".fits"
        logger.info("Writing FITS output to %s" % (out_fn))

        extlist = [fits.PrimaryHDU()]

        for i, part in enumerate(['BEST', 'WEIGHTED', 'SUM']):
            extlist.append(
                fits.ImageHDU(data=spectra_1d[:,:,i].T,
                              name="SCI.%s" % (part),)
            )
            extlist.append(
                fits.ImageHDU(data=variance_1d[:, :, i].T,
                              name="VAR.%s" % (part), )
            )

        # create the output multi-extension FITS
        # spec_3d = numpy.empty((2, spectra_1d.shape[1], spectra_1d.shape[0]))
        # spec_3d[0, :, :] = spectra_1d.T[:, :]
        # spec_3d[1, :, :] = variance_1d.T[:, :]

        # numpy.append(spectra_1d.reshape((spectra_1d.shape[1], spectra_1d.shape[0], 1)),
        #                    variance_1d.reshape((spectra_1d.shape[1], spectra_1d.shape[0], 1)),
        #                     axis=2,
        #                    )
        # print spec_3d.shape
        hdulist = fits.HDUList(extlist)
        # [
        #     fits.PrimaryHDU(header=hdu[0].header),
        #     fits.ImageHDU(data=spectra_1d.T, name="SCI"),
        #     fits.ImageHDU(data=variance_1d.T, name="VAR"),
        #     fits.ImageHDU(data=spec_3d, name="SPEC3D")
        # ])
        # add headers for the wavelength solution
        for ext in hdulist[1:]: #['SCI', 'VAR']:
            ext.header['WCSNAME'] = "calibrated wavelength"
            ext.header['CRPIX1'] = 1.
            ext.header['CRVAL1'] = wl0
            ext.header['CD1_1'] = dwl
            ext.header['CTYPE1'] = "AWAV"
            ext.header['CUNIT1'] = "Angstrom"
            for i, yr in enumerate(y_ranges):
                keyname = "YR_%03d" % (i + 1)
                value = "%04d:%04d" % (yr[0], yr[1])
                ext.header[keyname] = (value, "y-range for aperture %d" % (i + 1))
        hdulist.writeto(out_fn_fits, clobber=True)
        logger.info("done writing results (%s)" % (out_fn_fits))

    if ("ascii" in output_format):
        out_fn_ascii = out_fn + '.dat'
        out_fn_asciivar = out_fn + '.var'
        logger.info("Writing output as ASCII to %s / %s" % (out_fn_ascii,
                                                            out_fn_asciivar))

        with open(out_fn_ascii, "w") as of:
            for aper, yr in enumerate(y_ranges):
                print >>of, "# APERTURE: ", yr
                numpy.savetxt(of, numpy.append(out_wl.reshape((-1, 1)),
                                               spectra_1d[:,aper,:],
                                               axis=1
                                               )
                              )
                print >>of, "\n"*5

        with open(out_fn_asciivar, "w") as of:
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
        logger.info("done writing ASCII results")



    print brightest[0]
    pysalt.mp_logging.shutdown_logging(log_setup)


#!/usr/bin/env python

import os, sys, pyfits
import numpy
from scipy.ndimage.filters import median_filter
import bottleneck
import scipy.interpolate
numpy.seterr(divide='ignore', invalid='ignore')

# Disable nasty and useless RankWarning when spline fitting
import warnings
warnings.simplefilter('ignore', numpy.RankWarning)
# also ignore some other annoying warning
warnings.simplefilter('ignore', RuntimeWarning)

import bottleneck


import scipy.spatial
import pysalt.mp_logging
import logging
from optparse import OptionParser


import matplotlib.pyplot as pl


def extract_spectra(filename, extname, specs):

    hdulist = pyfits.open(filename)

    wl_data = hdulist['WAVELENGTH'].data
    obj_data = None
    try:
        obj_data = hdulist[extname].data
        logger.info("Using data from %s" % (extname))
    except:
        logger.error("Can not access/find extension %s" % (extname))
        return

    var_data = None
    var_valid = False
    try:
        var_data = hdulist['VAR'].data
        var_valid = True
    except:
        pass

    for ispec, spec in enumerate(specs):
        try:
            items = spec.split(",")
            from_line = int(items[0])
            to_line = int(items[1])
            output_fn = items[2]
            logger.info("Extracting lines %d -- %d to %s" % (
                from_line, to_line, output_fn
            ))
        except:
            logger.error("Error processing parameter %d: %s" % (ispec+1, spec))
            continue

        #
        # Get wavelength scale
        #
        # spec_wl = numpy.average(
        spec_wl = wl_data[from_line:to_line + 1, :]  # , axis=1)
        spec_wl_1d = numpy.average(spec_wl, axis=0)
        print spec_wl.shape, spec_wl_1d.shape

        #
        # Also extract fluxes
        #
        spec_fluxes = obj_data[from_line:to_line + 1, :]
        spec_fluxes_1d = numpy.sum(spec_fluxes, axis=0)
        print spec_fluxes_1d.shape

        #
        # Compute variance data (this is noise^2), so need to take sqrt to get real noise
        #
        var_1d = numpy.zeros((spec_wl_1d.shape[0]))
        if (var_valid):
            var_2d = var_data[from_line:to_line + 1, :]
            var_1d = numpy.sqrt(numpy.sum(var_2d, axis=0))
            print "Found VAR extension"

        #
        # Now merge fluxes and wavelength scale
        #
        combined = numpy.zeros((spec_wl_1d.shape[0], 4))
        combined[:, 0] = numpy.arange(combined.shape[0]) + 1
        combined[:, 1] = spec_wl_1d[:]
        combined[:, 2] = spec_fluxes_1d[:]
        combined[:, 3] = var_1d[:]

        # numpy.append(spec_wl_1d.reshape((-1,1)),
        #                             spec_fluxes_1d.reshape((-1,1)),
        #                             axis=1)
        print combined.shape

        numpy.savetxt(
            output_fn,
            combined,
                      header="""\
Column  1: x-coordinate [pixels]
Column  2: wavelength [angstroems]
Column  3: integrated flux [ADU]
Column  4: variance of sum (=sqrt(sum_i(var_i))) [ADU]
--------------------------------------------------------\
""",
                      )


if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()
    logger = logging.getLogger("SpecExtract")

    parser = OptionParser()
    parser.add_option("-e", "--ext", dest="extname",
                      help="name of extension to be extracted from",
                      default="SKYSUB.OPT")
    (options, cmdline_args) = parser.parse_args()

    filename = cmdline_args[0]

    extract_spectra(
        filename=filename,
        extname=options.extname,
        specs=cmdline_args[1:],
    )


    pysalt.mp_logging.shutdown_logging(logger_setup)

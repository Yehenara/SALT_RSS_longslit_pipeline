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

    print "unpickling input data"
    (width, supersample, source_profile_2d) = pickle.load(open(fn, "rb"))

    print "creating optimal weight class"
    optimal_weight = optimal_extraction.integrate_source_profile(
        width=width,
        supersample=supersample,
        profile2d=source_profile_2d,
        wl_resolution=-5,
    )

    print "sampling weight distribution"

    wl_range = [5500, 7000]
    y_range = [-50,50]

    n_random = 1000
    #random_coords = numpy.random.random((n_random,2)) * [(wl_range[1]-wl_range[0]), (y_range[1]-y_range[0])] + [wl_range[0], y_range[0]]
    #numpy.savetxt("random_test", random_coords)

    spec_wl_data = numpy.random.random((n_random)) * (wl_range[1]-wl_range[0]) + wl_range[0]
    spec_y_data = numpy.random.random((n_random)) * (y_range[1]-y_range[0]) + y_range[0]
    weight_data = numpy.zeros_like(spec_wl_data)
    weight_data = optimal_weight.get_weight(
                wl=spec_wl_data, y=spec_y_data)

    combined = numpy.array([spec_wl_data, spec_y_data, weight_data])
    print combined.shape
    numpy.savetxt("random_test2", combined.T)

    pysalt.mp_logging.shutdown_logging(log_setup)

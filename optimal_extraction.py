#!/usr/bin/env python


import os
import sys
import astropy.io.fits as fits
import numpy

import find_sources
import tracespec



def generate_source_profile(data, variance, trace_offset,
                            position, width=50):

    # get coordinates
    y,x = numpy.indices(data.shape, dtype=numpy.float)

    pos_x, pos_y = position[0], position[1]

    print y.shape
    print trace_offset.shape

    traceoffset2d = numpy.repeat(trace_offset.reshape((1,-1)), data.shape[0], axis=0)
    print traceoffset2d.shape

    dy = y - pos_y #+ traceoffset2d
    fits.PrimaryHDU(data=dy).writeto("optex_dy.fits", clobber=True)

    in_range = numpy.fabs(dy) < width+0.5
    src_data = data[in_range]
    src_var = variance[in_range]
    src_x = x[in_range]
    src_y = dy[in_range]

    combined = numpy.empty((src_data.shape[0], 4))
    combined[:,0] = src_data
    combined[:,1] = src_var
    combined[:,2] = src_x
    combined[:,3] = src_y
    numpy.savetxt("optex.data", combined)

    return



if __name__ == "__main__":

    fn = sys.argv[1]
    print fn





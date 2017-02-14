#!/usr/bin/env python


import os
import sys
import astropy.io.fits as fits
import numpy
import pysalt.mp_logging
import scipy.interpolate
import scipy.optimize

import wlcal
import traceline

import logging


def mirror_residuals(p, xmin, xmax, linefit, save):
    # check how far out we can go
    _left = p[0] - xmin
    _right = xmax - p[0]
    max_size = numpy.min([_left, _right])
    right = linefit(numpy.arange(max_size, dtype=numpy.float) + p[0])
    left = linefit(numpy.arange(-max_size, 0, dtype=numpy.float) + p[0])[::-1]
    diff = right - left
    ret = numpy.var(diff)
    if (save):
        numpy.savetxt("symmetry.residuals%d" % (p[0]), diff)
        with open("fitlog","a") as fl:
            fl.write("%f %f\n" % (p[0], ret))
    return ret



def find_curvature_symmetry_line(hdulist,
                                 data_ext='VAR',
                                 avg_width=10,
                                 n_lines=3,
                                 debug=True):

    logger = logging.getLogger("FindSymmetryRow")
    logger.info("Search for symmetry row, using %d lines (avg:%d)" % (
        n_lines, avg_width
    ))

    w = 10
    data = hdulist[data_ext].data

    logger.debug("extracting raw spectrum")
    #avg_width = 10
    spec = wlcal.extract_arc_spectrum(hdulist,
                                      avg_width=avg_width)
    numpy.savetxt("symmetry.spec", spec)

    logger.debug("Searching for lines")
    linelist = wlcal.find_list_of_lines(
        spec=spec,
        readnoise=2,
        gain=1,
        avg_width=avg_width,
        pre_smooth=None)
    #print linelist
    numpy.savetxt("symmetry.lines", linelist)

    #
    # Select the brightest line to determine line width
    #
    logger.debug("Picking brightest line")
    i_brightest = numpy.argmax(linelist[:,1])
    brightest = linelist[i_brightest,:]
    print brightest

    # allow for at most 20 pixels linewidth
    maxw=10
    part_of_bright_line = spec[int(brightest[0]-maxw):int(brightest[0]+maxw)]
    numpy.savetxt("symmetry.partofbrightest", part_of_bright_line)
    peak_flux = brightest[1]
    _x = numpy.arange(part_of_bright_line.shape[0])
    left = numpy.min(_x[part_of_bright_line > 0.5*peak_flux])
    right = numpy.max(_x[part_of_bright_line > 0.5*peak_flux])
    # print left, right, right-left
    linewidth = right-left
    if (linewidth < 3): linewidth=3
    logger.debug("Found linewidth: %d pixels" % (linewidth))

    logger.debug("Searching for lines again, now accounting for linewidths")
    linelist2 = wlcal.find_list_of_lines(
        spec=spec,
        readnoise=2,
        gain=1,
        avg_width=avg_width,
        pre_smooth=linewidth/2.)
    #print linelist
    numpy.savetxt("symmetry.lines2", linelist2)

    #
    # Now pick some isolated lines
    #
    i_isolated = traceline.pick_line_every_separation(
        arc_linelist=linelist2,
        trace_every=0.05,
        min_line_separation=3*linewidth,
        n_pixels=data.shape[1],
        min_signal_to_noise=10,
    )
    isolated = linelist2[i_isolated.astype(numpy.int)]
    numpy.savetxt("symmetry.isolated", isolated)
    print linelist2.shape, isolated.shape

    #
    # Sort the list of isolated lines by s/n
    #
    s2n_sort = numpy.argsort(isolated[:,4])[::-1]
    lines4curvature = isolated[s2n_sort[:n_lines]]

    numpy.savetxt(sys.stdout, lines4curvature, "%.1f")

    #
    #
    #
    wls_data = dict(
        linelist_arc=lines4curvature,
        line=int(data.shape[0]/2.),
    )

    symmetry_line = numpy.empty((n_lines,3))

    for line in range(lines4curvature.shape[0]):

        line_x = lines4curvature[line,0]
        logger.info("Finding symmetry point for ARC line at x=%d" % (line_x))
        lt = traceline.trace_single_line(
            fitsdata=data.T.copy(),
            wls_data=wls_data,
            line_idx=line,
            fine_centroiding=True,
            fine_centroiding_width=2*linewidth,
        )
        numpy.savetxt("symmetry.lt.%d" % (lines4curvature[line,0]), lt)

        #
        # Now we have the line-trace, including fine-tracing.
        # Next step: Reject outliers that can mess things up, as well as
        # noisy regions
        #
        blocksize=10
        to_pad = blocksize - lt.shape[0]%blocksize
        #print lt.shape, blocksize, to_pad
        pad_left = int(to_pad/2)
        pad_right = to_pad - pad_left
        lt_padded = numpy.pad(
            array=lt[:,4], pad_width=((pad_left, pad_right)),
            mode='constant', constant_values=numpy.NaN,
        )
        #print lt_padded.shape

        pos_2d = numpy.reshape(lt_padded, (-1, blocksize))
        #print pos_2d.shape
        var = numpy.var(pos_2d, axis=1)
        #print var.shape
        numpy.savetxt("symmetry.posvar", var)

        _var = var.copy()
        n_sigma = 5
        for iter in range(3):
            _stats = numpy.nanpercentile(_var, [16,50,84])
            _median = _stats[1]
            _sigma = 0.5*(_stats[2]-_stats[0])
            #print _stats, _median, _sigma
            bad = (_var > _median+n_sigma*_sigma) | (_var < _median - n_sigma*_sigma)
            _var[bad] = numpy.NaN
        numpy.savetxt("symmetry.posvar2", _var)
        bad_positions = numpy.isnan(_var) #.reshape((-1,1))
        #print bad_positions.shape
        idx = numpy.arange(-pad_left, lt.shape[0]+pad_right).reshape((-1, blocksize))
        #print "IDX/BAD:", idx.shape, bad.shape
        idx[bad_positions] = -1
        #print idx

        # extract only parts of the line profile not found to be bad
        bad1d = numpy.reshape(idx, (-1))[pad_left:-pad_right]
        #print bad1d.shape
        #print bad1d

        good_linedata = lt[bad1d[bad1d>=0]]
        numpy.savetxt("symmetry.good_lt.%d" % (lines4curvature[line,0]),
                      good_linedata)

        #
        #
        #

        linefit = scipy.interpolate.interp1d(
            x=good_linedata[:,0],
            y=good_linedata[:,4],
            kind='linear',
            bounds_error=False,
            fill_value=numpy.NaN,
        )

        xmin = numpy.min(good_linedata[:, 0])
        xmax = numpy.max(good_linedata[:, 0])

        # test fitting
        fl = open("fitlog_%d" %(line),"w")
        t1 = int(0.45*data.shape[0])
        t2 = int(0.55*data.shape[0])
        symmetry_quality = numpy.empty((data.shape[0]))
        symmetry_quality[:] = 1e9 #numpy.NaN
        for p in range(t1,t2):
            diff = mirror_residuals(
                p=[p],
                xmin=numpy.min(good_linedata[:, 0]),
                xmax=numpy.max(good_linedata[:, 0]),
                linefit=linefit,
                save=True)
            #numpy.savetxt("symmetry.residuals%d" % (p), diff)
            #print p, numpy.mean(diff), numpy.var(diff)
            print >>fl, p, diff
            symmetry_quality[p] = diff

        numpy.savetxt("symmetry.quality.%d" % (line), symmetry_quality)
            #numpy.mean(diff), numpy.var(diff)

        midline = numpy.argmin(symmetry_quality)
        best_var = symmetry_quality[midline]
        #print "MIDLINE:", midline

        # opt = scipy.optimize.leastsq(
        #     func=mirror_residuals,
        #     x0=[data.shape[0]/2.],
        #     args=(xmin, xmax, linefit, True),
        # )
        # print opt

        symmetry_line[line] = [lines4curvature[line,0], midline, best_var]

    #
    # Now we have all results, collect them into a single value as answer
    #
    good_lines = numpy.isfinite(symmetry_line[:,2])
    good_symmetry = symmetry_line[good_lines]
    best_match = numpy.argmin(good_symmetry[:,2])
    best_midline = good_symmetry[best_match]

    numpy.savetxt("symmetry.summary", symmetry_line)

    logger.info("Done finding symmetry (row=%d)" % (best_midline[1]))
    return symmetry_line, best_midline, linewidth


if __name__ == "__main__":

    logger = pysalt.mp_logging.setup_logging()


    fn = sys.argv[1]
    hdulist = fits.open(fn)

    data = hdulist['VAR'].data

    symmetry_lines, best_midline, linewidth = find_curvature_symmetry_line(
        hdulist = hdulist,
        data_ext='VAR',
        avg_width=5,
        n_lines=15,
    )

    print symmetry_lines
    numpy.savetxt("symmetry.summary", symmetry_lines)

    print best_midline

    pysalt.mp_logging.shutdown_logging(logger)


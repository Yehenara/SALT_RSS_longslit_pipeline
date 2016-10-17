#!/usr/bin/env python

import os, sys, numpy, scipy, pyfits
import logging

import pysalt
import traceline
import prep_science

def compute_spectrum_trace(data, start_x, start_y, xbin=1):

    # transpose - we can only trace up/down, not left/right
    # data = data.T

    # combined = traceline.trace_arc(
    #     data=data,
    #     start=start,
    #     direction=-1,
    #     )
    # print combined

    w = 30

    _y = start_y

    iy, ix = numpy.indices(data.shape)
    positions = numpy.empty((data.shape[1]))
    positions[:] = numpy.NaN

    # forward and backwards
    start_stop = [(start_x, data.shape[1], +1*xbin, start_y),
                  (start_x, -1, -1*xbin, start_y)]


    for dir in start_stop:
        _start, _end, _step, _start_y = dir
        for x in range(_start, _end, _step):
            # print x

            slit = data[_y - w:_y + w, x:x+xbin]
            pos_y = iy[_y - w:_y + w, x:x+xbin]

            valid = numpy.isfinite(slit)
            # print x, valid

            if (valid.any()):
                weighted_y = numpy.sum((slit * pos_y)[valid]) / numpy.sum(slit[valid])
            else:
                weighted_y = numpy.NaN

            if (weighted_y < 0):
                weighted_y = numpy.NaN

            # print slit.shape

            # print x, weighted_y

            numpy.savetxt("slit_%04d" % (x),
                          numpy.append(pos_y.reshape((-1, 1)),
                                       slit.reshape((-1, 1)), axis=1))

            positions[x] = weighted_y

    pos_x = numpy.arange(positions.shape[0])

    #
    # Now filter the profile to create a smoother trace profile
    #
    smoothtrace = prep_science.compute_smoothed_profile(
        data_x=pos_x.copy(),
        data_y=positions.copy(),
        #        n_max_neighbors=100,
        avg_sample_width=100,
        n_sigma=3,
    )

    combined = numpy.empty((positions.shape[0], 3))
    combined[:,0] = pos_x
    combined[:,1] = positions[:]
    combined[:,2] = smoothtrace[:]
    #numpy.savetxt("tracespec", positions)
    #numpy.savetxt("tracespec.smooth", smoothtrace)

    numpy.savetxt("tracespec", combined)

    return combined



def compute_trace_slopes(tracedata, n_iter=3, polyorder=1):

    npixels = tracedata.shape[0] #tracedata[-1,0]
    detector_size = npixels / 3

    poly_fits = [None]*3
    for detector in range(3):

        x_start = detector * detector_size
        x_end = x_start + detector_size

        in_detector_mask = (tracedata[:,0] >= x_start) & (tracedata[:,0] <= x_end) & (numpy.isfinite(tracedata[:,1]))
        _detector_trace = tracedata[in_detector_mask]

        # make space for the best-fit solutions of each iteration
        detector_trace = numpy.empty((_detector_trace.shape[0], _detector_trace.shape[1]+n_iter))
        detector_trace[:, :_detector_trace.shape[1]] = _detector_trace

        # detector_trace = numpy.append(tracedata[in_detector_mask],
        #                               tracedata[in_detector_mask][:,0:1], axis=1)

        valid = numpy.isfinite(detector_trace[:,0]) & numpy.isfinite(detector_trace[:,1])
        for iteration in range(n_iter):

            try:
                poly = numpy.polyfit(
                    x=detector_trace[:,0][valid],
                    y=detector_trace[:,1][valid],
                    deg=polyorder,
                )
            except Exception as e:
                print e
                break

            fit = numpy.polyval(poly, detector_trace[:,0])
            residual = detector_trace[:,1] - fit
            _perc = numpy.nanpercentile(residual[valid], [16,84,50])
            _med = _perc[2]
            _sigma = 0.5*(_perc[1] - _perc[0])
            #print _sigma, _perc

            bad = (residual > 3*_sigma) | (residual < -3*_sigma)
            valid[bad] = False

            detector_trace[:,iteration+_detector_trace.shape[1]] = fit

        poly_fits[detector] = poly

        print "Detector %d: %s" % (detector+1, str(poly))
        numpy.savetxt("det_trace.%d" % (detector + 1), detector_trace)

    # Now compensate all slopes to be offsets relative to the trace at the center of the detector
    poly_fits = numpy.array(poly_fits)
    numpy.savetxt(sys.stdout, poly_fits)

    mid_x = 0.5 * npixels
    center_y = numpy.polyval(poly_fits[1], mid_x)
    print center_y

    trace_offset = numpy.zeros((npixels))
    for detector in range(3):

        x_start = detector * detector_size
        x_end = x_start + detector_size

        tracepos = numpy.polyval(poly_fits[detector], numpy.arange(npixels))
        trace_offset[x_start:x_end] = tracepos[x_start:x_end] - center_y

    numpy.savetxt("tracespec.offset", trace_offset)
    return poly_fits, trace_offset


if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()
    logger = logging.getLogger("MAIN")

    hdulist = pyfits.open(sys.argv[1])
    
    start_x = int(sys.argv[2])-1
    start_y = int(sys.argv[3])-1
    try:
        xbin = int(sys.argv[4])
    except:
        xbin = 1

    start = start_x,start_y

    data = hdulist['SKYSUB.OPT'].data
    pyfits.PrimaryHDU(data=data).writeto("tracespec.fits", clobber=True)

    print "computing spectrum trace"
    spectrace_data = compute_spectrum_trace(data=data, start_x=start_x, start_y=start_y, xbin=xbin)

    print "finding trace slopes"
    slopes, trace_offset = compute_trace_slopes(spectrace_data)

    # Now generate a ds9 region file illustrating the actual peak positions, the best-fit lines and
    # 2 lines on either side to show the slit extent
    print "writing ds9 region file"
    with open("tracespec.reg", "w") as ds9:
        # write header
        print >>ds9, """# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
physical"""

        # write all points
        print >>ds9, "\n".join(["point(%.2f,%.2f) # point=x" % (d[0]+1, d[1]+1) for d in spectrace_data[numpy.isfinite(spectrace_data[:,1])]])

        slitwidth = 10
        #line = trace_offset + start_x + 10
        for slitwidth in [+10, -10, +30, -30]:
            print >>ds9, "\n".join([
                "line(%.2f,%.2f,%.2f,%.2f) # line=0 0" % (i+1, trace_offset[i]+start_y+slitwidth+1,
                                                          i+1+1, trace_offset[i+1]+start_y+slitwidth+1) for i in range(trace_offset.shape[0]-1)
            ])
    pysalt.mp_logging.shutdown_logging(logger_setup)

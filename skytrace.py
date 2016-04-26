#!/usr/bin/env python

import pyfits
import os, sys
import numpy
import scipy
import traceline
import prep_science
import pysalt.mp_logging
import wlcal
import bottleneck
import traceline

def trace_full_line(imgdata, x_start, y_start, window=5):

    weighted_pos = numpy.zeros((imgdata.shape[0],4))
    weighted_pos[:,:] = numpy.NaN

    x_start = int(x_start)

    x_guess_list = []
    x_guess = x_start

    x_pos_all = numpy.arange(imgdata.shape[1])
    for y in range(y_start, imgdata.shape[0]):
       
        # compute center of line in this row
        if (x_guess-window < 0 or
            x_guess+window >= imgdata.shape[1]):
            continue

        select = (x_pos_all >= x_guess-window) & (x_pos_all <= x_guess+window+1)
        x_pos = x_pos_all[select] #numpy.arange(x_guess-window, x_guess+window+1)
        try:
            flux = imgdata[y, select] #x_guess-window:x_guess+window+1]
        except:
            print x_guess, window, y
            break
            continue

        #print flux.shape, x_pos.shape
        i_flux = numpy.sum(flux)
        _wp = numpy.sum(x_pos*flux) / i_flux

        x_guess_list.append(_wp)
        
        x_guess = numpy.median(numpy.array(x_guess_list[-5:]))
        weighted_pos[y,:] = [y, _wp, x_guess, i_flux]
        #print y,_wp,x_guess

    x_guess = x_start
    x_guess_list = []
    for y in range(y_start, 0, -1):
       
        if (x_guess-window < 0 or
            x_guess+window >= imgdata.shape[1]):
            continue

        # compute center of line in this row
        select = (x_pos_all >= x_guess-window) & (x_pos_all <= x_guess+window+1)
        x_pos = x_pos_all[select] #numpy.arange(x_guess-window, x_guess+window+1)
        try:
            flux = imgdata[y, select] #x_guess-window:x_guess+window+1]
        except:
            print x_guess, window, y
            break
            continue

        #print flux.shape, x_pos.shape
        i_flux = numpy.sum(flux)
        _wp = numpy.sum(x_pos*flux) / i_flux
        # x_pos = numpy.arange(x_guess-window, x_guess+window+1)
        # try:
        #     flux = imgdata[y, x_guess-window:x_guess+window+1]
        # except:
        #     print x_guess, window, y
        #     break
        #     continue
        # i_flux = numpy.sum(flux)
        # _wp = numpy.sum(x_pos*flux) / i_flux


        x_guess_list.append(_wp)
        
        x_guess = numpy.median(numpy.array(x_guess_list[-5:]))
        weighted_pos[y,:] = [y, _wp, x_guess, i_flux]
        #print y,_wp,x_guess

    return weighted_pos

if __name__ == "__main__":

    logger = pysalt.mp_logging.setup_logging()

    fn = sys.argv[1]
    hdulist = pyfits.open(fn)

    # imgdata = hdulist['SCI.RAW'].data
    try:
        imgdata = hdulist['SCI.NOCRJ'].data
    except:
        imgdata = hdulist['SCI/'].data

    skylines, continuum = prep_science.filter_isolate_skylines(data=imgdata)
    pyfits.PrimaryHDU(data=skylines).writeto("skytrace_sky.fits", clobber=True)
    pyfits.PrimaryHDU(data=continuum).writeto("skytrace_continuum.fits", clobber=True)

    # pick a region close to the center, extract block of image rows, and get 
    # line list
    sky1d = bottleneck.nanmean(imgdata[550:575, :].astype(numpy.float32), axis=0)
    print sky1d.shape
    sky_linelist = wlcal.find_list_of_lines(sky1d, avg_width=25, pre_smooth=None)
    numpy.savetxt("sky1d", sky1d)
    numpy.savetxt("skylines.all", sky_linelist)

    # select lines with good spacing
    good_lines = traceline.pick_line_every_separation(
        arc_linelist=sky_linelist,
        trace_every=0.02,
        min_line_separation=0.01,
        n_pixels=imgdata.shape[1],
        min_signal_to_noise=7,
        )
    numpy.savetxt("skylines.good", sky_linelist[good_lines])

    print "X",skylines.shape, sky_linelist.shape, good_lines.shape
    selected_lines = sky_linelist[good_lines]
    print "selected:", selected_lines.shape

    all_traces = []

    linetraces = open("skylines.traces", "w")
    for idx, pick_line in enumerate(selected_lines):
        print pick_line

        wp = trace_full_line(skylines, x_start=pick_line[0], y_start=562, window=5)
        numpy.savetxt(linetraces, wp)
        print >>linetraces, "\n"*5
        all_traces.append(wp)

    numpy.savetxt("skylines.picked", selected_lines)
    for idx in range(selected_lines.shape[0]):
        pick_line = selected_lines[idx,:]
        #print pick_line

    all_traces = numpy.array(all_traces)
    print all_traces.shape

    ##########################################################################
    #
    # Now do some outlier rejection
    #
    ##########################################################################

    #
    # Compute average profile shape and mean intensity profile
    #
    _cl, _cr = int(0.4*all_traces.shape[1]), int(0.6*all_traces.shape[1])
    central_position = numpy.median(all_traces[:,_cl:_cr,:], axis=1)
    numpy.savetxt("skytrace_median", central_position)
    print central_position

    # subtract central position
    all_traces[:,:,1] -= central_position[:,1:2]
    all_traces[:,:,2] -= central_position[:,2:3]

    # scale intensity by median flux
    all_traces[:,:,3] /= central_position[:,3:]

    with open("skylines.traces.norm", "w") as lt2:
        for line in range(all_traces.shape[0]):
            numpy.savetxt(lt2, all_traces[line,:,:])
            print >>lt2, "\n"*5

    #
    # Do the spatial outlier correction first
    #
    profiles = all_traces[:,:,1]
    print profiles.shape
    for iteration in range(3):
        quantiles = scipy.stats.scoreatpercentile(
            a=profiles, 
            per=[16,50,84],
            axis=0,
            limit=(-1*all_traces.shape[1], 2*all_traces.shape[1])
            )
        median = quantiles[1]
        print median

        sigma = 0.5*(quantiles[2] - quantiles[0])

        outlier = (profiles > median+3*sigma) | (profiles < median-3*sigma)
        profiles[outlier] = numpy.NaN

    
    with open("skylines.traces.clean", "w") as lt2:
        for line in range(all_traces.shape[0]):
            numpy.savetxt(lt2, all_traces[line,:,:])
            print >>lt2, "\n"*5

    medians = bottleneck.nanmedian(all_traces, axis=0)
    numpy.savetxt("skylines.traces.median", medians)
    print medians.shape

    stds = bottleneck.nanstd(all_traces, axis=0)
    stds[:,0] = medians[:,0]
    numpy.savetxt("skylines.traces.std", stds)

    #
    # Now reconstruct the final line traces, filling in gaps with values 
    # predicted by the median profile
    #
    all_median = numpy.repeat(medians.reshape((-1,1)), all_traces.shape[0], axis=1)
    print all_median.shape, all_traces[:,:,1].shape
    outlier = numpy.isnan(all_traces[:,:,1])
    print outlier.shape
    print outlier
    try:
        all_traces[:,:,1][outlier] = all_median[:,:][outlier]
    except:
        pass
    all_traces[:,:,1] += central_position[:,1:2]


    with open("skylines.traces.corrected", "w") as lt2:
        for line in range(all_traces.shape[0]):
            numpy.savetxt(lt2, all_traces[line,:,:])
            print >>lt2, "\n"*5

    with open("skylines.traces.corrected2", "w") as lt2:
        for line in range(all_traces.shape[0]):
            numpy.savetxt(lt2, all_median[:,:])
            print >>lt2, "\n"*5


        

    # combined = traceline.trace_arc(
    #     data=skylines,
    #     start=(601,526),
    #     #start=(526,602),
    #     direction=+1,
    #     )

    # weighted_pos = numpy.zeros((imgdata.shape[0],4))
    # weighted_pos[:,:] = numpy.NaN

    # x_guess = 601
    # window = 5
    # x_guess_list = []

    # for y in range(526, imgdata.shape[0]):
       
    #     # compute center of line in this row
    #     x_pos = numpy.arange(x_guess-window, x_guess+window+1)
    #     flux = imgdata[y, x_guess-window:x_guess+window+1]
    #     i_flux = numpy.sum(flux)
    #     _wp = numpy.sum(x_pos*flux) / i_flux


    #     x_guess_list.append(_wp)
        
    #     x_guess = numpy.median(numpy.array(x_guess_list[-5:]))
    #     weighted_pos[y,:] = [y, _wp, x_guess, i_flux]
    #     print y,_wp,x_guess

    # x_guess = 601
    # x_guess_list = []
    # for y in range(526, 0, -1):
       
    #     # compute center of line in this row
    #     x_pos = numpy.arange(x_guess-window, x_guess+window+1)
    #     flux = imgdata[y, x_guess-window:x_guess+window+1]
    #     i_flux = numpy.sum(flux)
    #     _wp = numpy.sum(x_pos*flux) / i_flux


    #     x_guess_list.append(_wp)
        
    #     x_guess = numpy.median(numpy.array(x_guess_list[-5:]))
    #     weighted_pos[y,:] = [y, _wp, x_guess, i_flux]
    #     print y,_wp,x_guess

    # numpy.savetxt("skytrace_arc.txt", weighted_pos)
    # #print combined
    # #numpy.savetxt("skytrace_arc.txt", combined)

    wp2 = trace_full_line(imgdata, x_start=601, y_start=526, window=5)
    numpy.savetxt("skytrace_arc2.txt", wp2)

    pysalt.mp_logging.shutdown_logging(logger)

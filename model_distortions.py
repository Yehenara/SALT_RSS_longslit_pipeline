#!/usr/bin/env python


import os
import sys
import numpy
from astropy.io import fits
from pysalt import mp_logging
import logging
import scipy
import scipy.interpolate
import scipy.ndimage
import math

import traceline
import wlmodel
import map_distortions
import pysalt

distmap_cols = traceline.linetrace_cols[:4]
distmap_cols.extend(
    ['X_FINE',
     'FLUX',
     'WL_PIXEL',
     'WL_FINE',
     'WL_OFFSET'
     ]
)
# print distmap_cols
distmap_colidx = {}
for idx,name in enumerate(distmap_cols):
    distmap_colidx[name] = idx


def map_wavelength_distortions(skyline_list, wl_2d, img_2d,
                               diff_2d=None, badrows=None, s2n_cutoff=5,
                               ref_row=None, linewidth=10,
                               max_distortion=2.,
                               debug=False,
                               primary_header=None, xbin=2, ybin=2, symmetry_row=None,
                               min_line_count=10,
                               distortion_method='trace',
                               ):

    logger = logging.getLogger("ModelDistortions")

    if (ref_row is None):
        ref_row = 0.4 * wl_2d.shape[0]
    ref_row = int(ref_row)
    logger.info("Using reference row: %d" % (ref_row))

    # print "        X      peak continuum   c.noise       S/N      WL/X"
    # print "="*59
    # numpy.savetxt(sys.stdout, skyline_list, "%9.3f")
    # print "=" * 59

    # good_lines = numpy.isfinite(skyline_list[:,0]) & (skyline_list[:,4]>s2n_cutoff)
    good_lines = traceline.pick_line_every_separation(
        skyline_list,
        trace_every=5,
        min_line_separation=40,
        n_pixels=img_2d.shape[1],
        min_signal_to_noise=s2n_cutoff,
        )
    logger.debug("Selecting %d of %d lines to compute distortion model" % (
        good_lines.shape[0], skyline_list.shape[0]))
    skyline_list = skyline_list[good_lines]

    if (debug):
        print "\n"*5
        print "        X      peak continuum   c.noise       S/N      WL/X"
        print "="*59
        numpy.savetxt(sys.stdout, skyline_list, "%9.3f")
        print "=" * 59

    if (len(skyline_list) < min_line_count):
        return None, None

    #
    # Now load all files for these lines
    #
    avg_dispersion = (wl_2d[ref_row,-1] - wl_2d[ref_row,0]) / wl_2d.shape[1]
    d_wl = 1.5 * linewidth * avg_dispersion
    logger.info("Searching for WL distortion using a maximum tolerance of %.2f A" % (d_wl))

    # pre-filter the image data with the linewidth to make identifying line
    # centers easier and more accurate
    linewidth_sigma = linewidth / 2.3
    img_prefilter = scipy.ndimage.filters.gaussian_filter(
        input=img_2d,
        sigma=(0,linewidth_sigma),
        order=0,
        mode='reflect',
    )
    fits.PrimaryHDU(data=img_prefilter).writeto("dist_prefilter.fits", clobber=True)


    if (distortion_method.lower() == 'trace' or True):
        # for now this is the only working method

        #
        # Rather than the original re-centering, use the line-tracing
        # algorithm/method instead
        #
        multi_line_traces = []
        for line in skyline_list:

            logger.info("tracing line, starting at x=%d, y=%d" % (line[0], ref_row))
            # print line[0], ref_row

            all_row_data = None
            for direction_y in [-1,+1]:
                lt = traceline.trace_arc(
                    data=img_prefilter.T,
                    start=(line[0], int(ref_row)),
                    direction=direction_y,
                    max_window_x=linewidth,
                )

                valid = numpy.isfinite(lt[:,1])
                lt = lt[valid]

                # if (not ds9_region_file == None):
                #     print >>ds9_region, '# text(%d,%d) text={%d}' % (lt[0,1]+1, lt[0,0]+1, line_idx)

                #     for idx in range(1, lt.shape[0]):
                #         # print >>ds9_region, 'point(%d,%d)' % (lt[idx,1], lt[idx,0])
                #         print >>ds9_region, 'line(%d,%d,  %d,%d' % (lt[idx,1]+1, lt[idx,0]+1, lt[idx-1,1]+1, lt[idx-1,0]+1)

                all_row_data = lt if all_row_data is None else numpy.append(all_row_data, lt, axis=0)
            #
            # # Sort all_row_data by vertical position
            # si = numpy.argsort(all_row_data[:,0])
            # all_row_data = all_row_data[si]
            #
            # # with open("linetrace_idx.%d" % (line_idx), "w") as lt_file:
            # #     numpy.savetxt(lt_file, all_row_data)
            # #     print >>lt_file, "\n\n\n\n\n"
            #
            # # if (not ds9_region_file == None): ds9_region.close()
            #
            # #
            # # create a debug file for all line-traces combined
            # #
            #
            fine_centroiding = True
            fine_centroiding_width = linewidth
            if (fine_centroiding):
                logger.debug("Done with tracing, starting fine centroiding")
                #print all_row_data.shape

                # cutout regions close (+/- width pixels) to line
                # traced_y_pos = all_row_data[:,0].astype(numpy.int)
                # traced_x_pos = all_row_data[:,1].astype(numpy.int)
                # x1 = traced_x_pos - centroiding_width
                # x2 = traced_x_pos + centroiding_width+1
                # print x1
                # line_cutout = fitsdata[traced_y_pos, x1:x2]
                # print line_cutout.shape

                imghdu = fits.ImageHDU()
                fine_pos = traceline.subpixel_centroid_trace(
                    data=img_prefilter, tracedata=all_row_data,
                    width=fine_centroiding_width,
                    dumpfile=imghdu,
                )#"linetrace_%d.fits" % (line_idx))
                # if (not linetrace_hdulist == None):
                #     linetrace_hdulist.append(imghdu)
                #fits.PrimaryHDU(data=rectified).writeto("linetrace_%d.fits" % (line_idx), clobber=True)
                #print fine_pos
                fp1, fp2 = fine_pos

                #print fp1.shape, fp2.shape

            if (fp1.shape[0] != all_row_data.shape[0]):
                # something went wrong with the fine centroiding
                continue

            logger.debug("Found %d tracepoints" % (all_row_data.shape[0]))
            # print fp1.shape, fp2.shape, all_row_data.shape

            linetrace_prefinal = numpy.append(
                all_row_data,
                numpy.array([fp1, fp2]).T, axis=1)

            if (primary_header is not None):
                x_as_wl = wlmodel.rssmodelwave(
                    header=primary_header,
                    img=img_2d,
                    xbin=xbin, ybin=ybin,
                    y_center=symmetry_row,
                    x=linetrace_prefinal[:,[1,4]],
                    y=linetrace_prefinal[:,[0,0]],
                )
                # print x_as_wl.shape
            else:
                x_as_wl = linetrace_prefinal[:, [1,4]]

            linetrace_final = numpy.append(
                linetrace_prefinal, x_as_wl, axis=1
            )



            #
            #
            #
            #
            #
            # all_row_data += [1., 1., 0., 0.,]
            # numpy.savetxt("allrowdata.%d" % (line[0]), all_row_data)

            multi_line_traces.append(linetrace_final)

            linetrace_final += [1., 1., 0., 0., 1., 0., 0., 0.]
            numpy.savetxt("allrowdata.%d" % (line[0]), linetrace_final)
            numpy.savetxt("allrowdata.dum.%d" % (line[0]), linetrace_final[:, [0,1,4,-2,-1]])

        #
        # Now we have a full set of line-traces for all identified lines.
        # Go through the list, compute the mean wavelength close to the symmetry
        # point where curvature is at its lowest, and from that compute wavelength
        # shifts along the slit for each of the lines
        #
        # print multi_line_traces
        linetrace_combined = None

        y_range = 0.025 * img_2d.shape[1]
        symmetry_row_binned = symmetry_row / ybin
        binwidth = 20
        use_for_map = [False] * len(multi_line_traces)

        for i_line, linetrace in enumerate(multi_line_traces):

            logger.info("Computing wavelength distortion from line x=%d" % (skyline_list[i_line,0]))
            # print "\n\n\n"
            # print i_line, skyline_list[i_line,0], symmetry_row_binned
            # print linetrace[:,0]

            near_center = numpy.fabs(linetrace[:,distmap_colidx['Y']] - symmetry_row_binned) < y_range
            if (numpy.sum(near_center) > 10):
                mean_wl = numpy.mean(linetrace[:,distmap_colidx['WL_FINE']][near_center])
                wl_dispersion = numpy.std(linetrace[:,distmap_colidx['WL_FINE']][near_center])
                # print i_line, skyline_list[i_line,0], mean_wl, wl_dispersion

                wl_distortion = linetrace[:,distmap_colidx['WL_FINE']] - mean_wl
                use_for_map[i_line] = True
            else:
                logger.warning("No pixels close to symmetry line found")
                wl_distortion = linetrace[:,distmap_colidx['WL_FINE']] * numpy.NaN
                continue

            combined = numpy.append(
                linetrace, wl_distortion.reshape((-1,1)), axis=1
            )

            multi_line_traces[i_line] = combined

            numpy.savetxt("allrowdata.dist.%d" % (skyline_list[i_line,0]),
                          combined)


            #
            # Do some filtering based on mean positions and fluxes
            #
            logger.debug("Begin line filtering")
            n_to_add = int(binwidth - (combined.shape[0] % binwidth))
            # print combined.shape, binwidth, n_to_add
            n_add_front = int(math.ceil(n_to_add / 2))
            n_add_back = n_to_add - n_add_front
            padded = numpy.pad(
                array=combined,
                pad_width=((n_add_front,n_add_back),(0,0)),
                mode='constant',
                constant_values=(numpy.NaN,),
            ).reshape((-1, binwidth, combined.shape[1]))
            # print padded.shape
            combined_median = numpy.nanmedian(padded, axis=1)
            combined_var = numpy.nanvar(padded, axis=1)
            # print combined_median.shape, combined_var.shape

            # use pixels with small position variance to get a mean level
            # then select all pixels with proper fluxes as part of the trace
            small_pos_errors = combined_var[:,distmap_colidx['WL_FINE']] < avg_dispersion
            logger.debug("Selecting %d good pixels" % (numpy.sum(small_pos_errors)))
            flux_dist = numpy.nanpercentile(combined_median[:,distmap_colidx['FLUX']][small_pos_errors], [16,50,84])
            try:
                flux_median = flux_dist[1]
                flux_1sigma = 0.5*(flux_dist[2]-flux_dist[0])
            except:
                logger.critical("ERROR running stats: %s" % (str(flux_dist)))
                continue

            print "flux:", flux_median, flux_1sigma
            good_fluxes = (combined[:,distmap_colidx['FLUX']] > (flux_median-3*flux_1sigma)) & \
                          (combined[:,distmap_colidx['FLUX']] < (flux_median+3*flux_1sigma))
            good_trace = combined[good_fluxes]

            if (use_for_map[i_line]):
                numpy.savetxt("allrowdata.distgood.%d" % (skyline_list[i_line, 0]),
                              good_trace)
            else:
                numpy.savetxt("allrowdata.distbad.%d" % (skyline_list[i_line, 0]),
                              good_trace)

            if (use_for_map[i_line]):
                logger.debug("merging data")
                linetrace_combined = good_trace if linetrace_combined is None \
                    else numpy.append(linetrace_combined, good_trace, axis=0)

        #
        # Now we have a full set of datapoints with distortion values across the
        # detector. Continue to create the full-resolution, interpolated
        # wavelength distortion frame.
        #

        logger.debug("Computing 2-D interpolator")
        #wl_dist = x[numpy.isfinite(x[:,2])]
        wl_dist = linetrace_combined[:, [distmap_colidx['WL_FINE'],
                                         distmap_colidx['Y'],
                                         distmap_colidx['WL_OFFSET'],
                                         distmap_colidx['FLUX']
                                         ]]
        if (debug):
            numpy.savetxt("distortion_model.in", wl_dist)
        # print wl_dist.shape
        interpol = scipy.interpolate.SmoothBivariateSpline(
            x=wl_dist[:,0],
            y=wl_dist[:,1],
            z=wl_dist[:,2],
            #w=wl_dist[:,3],
            kx=3, ky=3,
        )

        if (debug):
            numpy.savetxt("interpol_in.x", wl_dist[:, 0])
            numpy.savetxt("interpol_in.y", wl_dist[:, 1])
            numpy.savetxt("interpol_in.z", wl_dist[:, 2])
            numpy.savetxt("interpol_in.w", wl_dist[:, 3])

        #
        # Now compute a full 2-d grid of distortions as fct. of y and wavelength positions
        #
        logger.debug("computing 2-d distortion model")
        wlmap = wl_2d #hdulist['WAVELENGTH'].data
        _y,_x = numpy.indices(wlmap.shape)
        distortion_2d = interpol(
            x=wlmap,
            y=_y,
            grid=False,)
        # print distortion_2d.shape


        # compute residuals
        model = interpol(x=wl_dist[:,0], y=wl_dist[:,1], grid=False)
        residuals = wl_dist[:,2] - model

        wl_dist_data = numpy.empty((wl_dist.shape[0], wl_dist.shape[1]+2))
        wl_dist_data[:, :wl_dist.shape[1]] = wl_dist
        wl_dist_data[:, -2] = model
        wl_dist_data[:, -1] = residuals

        if (debug):
            wl_dist[:,2] = model
            numpy.savetxt("distortion_model.out", wl_dist)
            wl_dist[:,2] = residuals
            numpy.savetxt("distortion_model.residuals", wl_dist)
            fits.PrimaryHDU(data=distortion_2d).writeto("wldist_raw.fits", clobber=True)

    distortion_2d[distortion_2d > max_distortion] = max_distortion
    distortion_2d[distortion_2d < -max_distortion] = -max_distortion

    return distortion_2d, wl_dist_data

    os._exit(0)
















    dist, dist_binned, bias_level, dist_median, dist_std  = \
        map_distortions.map_distortions(wl_2d=wl_2d,
                                        diff_2d=diff_2d,
                                        #img_2d=img_2d,
                                        img_2d=img_prefilter,
                                        x_list=skyline_list[:,0],
                                        y=ref_row, #img_2d.shape[0] / 2.
                                        badrows=badrows,
                                        dwl=d_wl,
                                        debug=debug,
        )
    # print len(dist)
    # print len(dist_binned)
    # print len(dist_median)
    # print len(dist_std)


    # print "\n----------"*10,"mapping distortions", "\n-------------"*5

    readnoise = 7
    gain=2

    regfile = open("distortions.reg", "w")
    print >>regfile, """\
# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
physical
    """

    all_lines = [] #[None] * skyline_list.shape[0]
    for i, line in enumerate(skyline_list):

        print >>regfile, "point(%.2f,%.2f) # point=circle" % (line[0], ref_row)

        #fn = "distortion_%d.bin" % (line[0])
        #linedata = numpy.loadtxt(fn)

        linedata_mean = dist_binned[i]
        linedata_med = dist_median[i]
        linedata_std = dist_std[i]

        # compute median s/n
        s2n = (linedata_med[:, 10] - linedata_med[:, 13]) / (
            numpy.sqrt((linedata_med[:, 13] * gain) + readnoise ** 2) / gain)
        median_s2n = numpy.median(s2n[s2n>0])
        logger.debug("Line @ %f --> median s/n = %f" % (line[0], median_s2n))

        # med_flux = numpy.median(linedata_mean[:, 11][numpy.isfinite(linedata_mean[:, 11])])
        # noise = numpy.sqrt(med_flux**2*gain + readnoise**2)/gain
        if (median_s2n < 4):
            logger.debug("Ignoring line at x=%f because of insuffient s/n" % (
                line[0]))
            continue

        #
        # Also consider the typical scatter in center positions
        #
        median_pos_scatter = numpy.nanmedian(linedata_std[:,8])
        logger.debug("Median position scatter: %f" % (median_pos_scatter))

        # compute the wavelength error from the actual line position
        linedata_mean[:, 9] -= linedata_mean[:, 7]

        # correct the line for a global error in wavelength
        not_nan = numpy.isfinite(linedata_mean[:, 9])
        med_dwl = numpy.median(linedata_mean[:, 9][not_nan])
        linedata_mean[:, 9] -= med_dwl

        # print "LINEDATA", i, "\n", linedata_mean, "\n\n"
        numpy.savetxt("linedata_%d" % (i), linedata_mean)

        #all_lines[i] = linedata_mean
        all_lines.append(linedata_mean)

    if (len(all_lines) <= 0):
        logger.warning("no lines found, aborting WL distortion modeling")
        return None, None

    all_lines = numpy.array(all_lines)

    logger.info("ALL-LINES SHAPE: %d,%d,%d" % (
        all_lines.shape[0], all_lines.shape[1], all_lines.shape[2]))
    try:
        wl_dist = all_lines[:, :, [7, 0, 9]] # wl,y,d_wl
        # print wl_dist.shape
        x = wl_dist.reshape((-1,wl_dist.shape[2]))
        # print x.shape
        if (debug):
            numpy.savetxt("distortion_model.in", x)
    except IndexError:
        logger.error("Index error when trying to create the distortion "
                     "model data (%s)" % (str(all_lines.shape)))
        return None, None


    #
    # Now convert all the data we have into a full 2-d model in wl & x
    #
    logger.debug("Computing 2-D interpolator")
    wl_dist = x[numpy.isfinite(x[:,2])]
    # print wl_dist.shape
    interpol = scipy.interpolate.SmoothBivariateSpline(
        x=wl_dist[:,0],
        y=wl_dist[:,1],
        z=wl_dist[:,2],
        kx=3, ky=3,
    )

    if (debug):
        numpy.savetxt("interpol_in.x", wl_dist[:,0])
        numpy.savetxt("interpol_in.y", wl_dist[:,1])
        numpy.savetxt("interpol_in.z", wl_dist[:,2])

    #
    # Now compute a full 2-d grid of distortions as fct. of y and wavelength positions
    #
    logger.debug("computing 2-d distortion model")
    wlmap = wl_2d #hdulist['WAVELENGTH'].data
    _y,_x = numpy.indices(wlmap.shape)
    distortion_2d = interpol(
        x=wlmap,
        y=_y,
        grid=False,)
    # print distortion_2d.shape


    # compute residuals
    model = interpol(x=wl_dist[:,0], y=wl_dist[:,1], grid=False)
    residuals = wl_dist[:,2] - model

    wl_dist_data = numpy.empty((wl_dist.shape[0], wl_dist.shape[1]+2))
    wl_dist_data[:, :wl_dist.shape[1]] = wl_dist
    wl_dist_data[:, -2] = model
    wl_dist_data[:, -1] = residuals

    if (debug):
        wl_dist[:,2] = model
        numpy.savetxt("distortion_model.out", wl_dist)
        wl_dist[:,2] = residuals
        numpy.savetxt("distortion_model.residuals", wl_dist)

    distortion_2d[distortion_2d > max_distortion] = max_distortion
    distortion_2d[distortion_2d < -max_distortion] = -max_distortion

    return distortion_2d, wl_dist_data




if __name__ == "__main__":

    logger_setup = mp_logging.setup_logging()


    fn = sys.argv[1]
    hdulist = fits.open(fn)

    img_size = hdulist['SCI'].header['NAXIS1']

    skytable_ext = hdulist['SKYLINES']

    n_lines = skytable_ext.header['NAXIS2']
    n_cols = skytable_ext.header['TFIELDS']
    skyline_list = numpy.empty((n_lines, n_cols))
    for i in range(n_cols):
        skyline_list[:,i] = skytable_ext.data.field(i)

    skylines_ref_y = 610
    if ('LINEREFY' in skytable_ext.header):
        skylines_ref_y = skytable_ext.header['LINEREFY']


    try:
        wl_2d = hdulist['WAVELENGTH'].data
    except:
        wl_2d = hdulist['WAVELENGTH.RAW'].data

    diff_2d = hdulist['SKYSUB.OPT'].data
    img_2d = hdulist['SCI'].data
    if ('SCI.CRJ' in hdulist):
        img_2d = hdulist['SCI.CRJ'].data

    try:
        badrows = hdulist['BADROWS'].data
        badrows = badrows > 0
    except:
        badrows = None

    linewidth = 4
    if ('LINEWDTH' in hdulist[0].header):
        linewidth = hdulist[0].header['LINEWDTH']

    xbin, ybin = pysalt.get_binning(hdulist)
    distortion_2d, dist_quality = map_wavelength_distortions(
        skyline_list=skyline_list,
        wl_2d=wl_2d,
        img_2d=img_2d,
        diff_2d=diff_2d,
        badrows=badrows,
        debug=True,
        linewidth=linewidth,
        ref_row=skylines_ref_y,
        primary_header=hdulist[0].header,
        xbin=xbin, ybin=ybin,
        symmetry_row=hdulist[0].header['RSSYCNTR'],
    )

    fits.PrimaryHDU(data=distortion_2d).writeto("distortion_2d.fits", clobber=True)
    numpy.savetxt("distortion_model.quality", dist_quality)

    mp_logging.shutdown_logging(logger_setup)





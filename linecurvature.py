#!/usr/bin/env python


import sys
import numpy
import scipy
import scipy.ndimage
import scipy.optimize
from astropy.io import fits
import matplotlib.pyplot

def linecenter(p, y):
    return p[0] + p[1]*y + p[2]*y**2

def err(p, data):

    x = data[:,1]
    wl = data[:,0]
    flux = data[:,3]
    y = data[:,2]

    distance_from_center = wl - linecenter(p,y)

    # now add weighting
    weighted = distance_from_center * flux

    return weighted


if __name__ ==  "__main__":

    print "linecurvature"

    full_fits_fn = sys.argv[1]
    print "loading fits frame to extract wavelength map"
    full_fits = fits.open(full_fits_fn)
    wavelength_map = full_fits['WAVELENGTH'].data

    for fn in sys.argv[2:]:

        print "\n\nloading data from", fn
        with open(fn, "rb") as datafile:
            data = numpy.load(datafile)
        #print data.shape

        mean_wavelength = numpy.mean(data[:,0])
        mean_column = numpy.mean(data[:,2])
        mean_x = numpy.median(data[:,1])

        y_min = numpy.min(data[:,2])
        y_max = numpy.max(data[:,2])
        y = numpy.arange(y_min, y_max)

        print "mean wavelength/column:", mean_wavelength, mean_column

        good_data = numpy.isfinite(data[:,3])

        p_init = [mean_wavelength, 0., 0.]

        for iteration in range(1):

            # fit a straight line
            fit = scipy.optimize.leastsq(
                func=err,
                x0 = p_init,
                args=(data[good_data]),
                full_output=1,
            )

            #print fit
            p_best = fit[0]
            print "current best fit:", p_best

        # remaining curvature
        linecenters = linecenter(p_best, y)
        #print "line centers vs y:", linecenters
        deviation = linecenters[0] - linecenters[-1]
            # numpy.max(linecenters) - numpy.min(linecenters)
        print "total deviation:", deviation, "angstroems"
        #print y

    #     #
    #     # find the data points closest to the best-fit at either end
    #     #
    #
    #
    #     for y_edge in [y_min, y_max]:
    #
    #         print "checking for x/y for y=",y_edge
    #         at_edge = (data[:, 2] == y_edge)
    #         data_edge = data[at_edge]
    #
    #         print "best-fit line wavelength:", linecenter(p_best,y_edge)
    #
    #         d_wl = numpy.fabs(data_edge[:,2] - linecenter(p_best,y_edge))
    #         closest = data_edge[numpy.argmin(d_wl)]
    #         print y_edge, closest
    #
    #

        # extract the relevant chunk from the wavelength map
        min_x = int(numpy.min(data[:,1]))
        max_x = int(numpy.max(data[:,1]))
        wl_chunk = wavelength_map[:, min_x:max_x+1]
        #print wl_chunk.shape

        # now compute the wavelength dispersion in y-direction
        wl_dispersion_y = numpy.diff(wl_chunk, axis=0)
        #print wl_dispersion_y.shape
        mean_wl_dispersion_y = numpy.mean(wl_dispersion_y, axis=1)
        #print mean_wl_dispersion_y.shape

        wl_dispersion_x = numpy.diff(wl_chunk, axis=1)
        mean_wl_dispersion_x = numpy.mean(wl_dispersion_x, axis=0)
        print "dispersion at edges, x-direction:", mean_wl_dispersion_x[0], mean_wl_dispersion_x[-1]

        #
        # now we can compute how much we have to shift the center to
        # compensate for the wavelength difference found above
        #
        # print "mean dispersion in y-dir:", mean_wl_dispersion_y
        edge_dispersion_per_pixel_shift = mean_wl_dispersion_y[0] - mean_wl_dispersion_y[-1]
        print "dispersion at edges (combined):", edge_dispersion_per_pixel_shift

        pixelshift = deviation / edge_dispersion_per_pixel_shift
        print "need shift of",pixelshift,"pixels"

        print "@@@", mean_wavelength, mean_column, mean_x, edge_dispersion_per_pixel_shift, pixelshift


        plot_fn = fn[:-4]+'.png'
        print "Creating plot", plot_fn
        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111)
        min_max = numpy.percentile(data[:,3], [1,99])
        print min_max
        ax.scatter(x=data[:,0], y=data[:,2], c=data[:,3], s=4, cmap='rainbow', edgecolors='face', vmin=min_max[0], vmax=min_max[1])
        ax.set_ylim((y_min,y_max))
        ax.set_xlim((numpy.min(data[:,0]), numpy.max(data[:,0])))
        ax.plot(linecenters, y, c='white', linewidth=5)
        ax.plot(linecenters, y, c='black', linewidth=2)
        #colorbar = matplotlib.pyplot.colorbar(cmap='viridis')
        #colorbar.set_label("phot. zeropoint")

        #ax.colorbar()
        fig.savefig(plot_fn)


        print "done!"
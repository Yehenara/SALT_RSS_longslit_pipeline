#!/usr/bin/env python


import os, sys
from astropy.io import fits
import numpy
import scipy
import scipy.interpolate
import math

from optimal_spline_basepoints import satisfy_schoenberg_whitney

from matplotlib import pyplot


def compute_good_spline(wl, flux, noise, n_bases):

    
    t_ref_points =  int(numpy.ceil((ref_wl_max - ref_wl_min) / basepoint_spacing)) \
                    if basepoint_spacing > 0 else int(numpy.fabs(basepoint_spacing))
    t_raw = numpy.linspace(ref_wl_min+3, ref_wl_max-3, t_ref_points)
    t_ref = numpy.empty(shape=(t_raw.shape[0]+4))
    t_ref[0] = t_raw[0]-2
    t_ref[1] = t_raw[0]-1
    t_ref[-2] = t_raw[-1]+1
    t_ref[-1] = t_raw[-1]+2
    t_ref[2:-2] = t_raw[:]
    print t_ref

    print "Fitting reference spline"
    ref_spline =  scipy.interpolate.LSQUnivariateSpline(
        x=ref[:,0], 
        y=ref[:,1], 
        t=t_raw, 
        w=ref[:,2], 
        #bbox=[None, None], 
        k=3)

    pass

if __name__ == "__main__":

    datafile = sys.argv[1]
    data = numpy.loadtxt(datafile)

    print "Load: %d datapoints from %.1f to %.1f A" % (
        data.shape[0], data[0,1], data[-1,1])

    # Select only good data points
    good_data = numpy.isfinite(data[:,1]) & numpy.isfinite(data[:,2])
    data = data[good_data]

    reference_file = sys.argv[2]
    ref = numpy.loadtxt(reference_file)

    print "done loading reference spectrum, wl. coverage: %.1f - %.1f A" % (
        ref[0,0], ref[-1,0])

    outfile = sys.argv[3]

    #
    #
    #
    basepoint_spacing = 50 # basepoint every 50 A
    basepoint_spacing = 100 # basepoint every 50 A
    ref_wl_min = ref[0,0]
    ref_wl_max = ref[-1,0]
    
    #
    # Compute spline for reference spectrum
    #
    t_ref_points =  int(numpy.ceil((ref_wl_max - ref_wl_min) / basepoint_spacing)) \
                    if basepoint_spacing > 0 else int(numpy.fabs(basepoint_spacing))
    t_raw = numpy.linspace(ref_wl_min+3, ref_wl_max-3, t_ref_points)
    t_ref = numpy.empty(shape=(t_raw.shape[0]+4))
    t_ref[0] = t_raw[0]-2
    t_ref[1] = t_raw[0]-1
    t_ref[-2] = t_raw[-1]+1
    t_ref[-1] = t_raw[-1]+2
    t_ref[2:-2] = t_raw[:]
    print t_ref

    print "Fitting reference spline"
    ref_spline =  scipy.interpolate.LSQUnivariateSpline(
        x=ref[:,0], 
        y=ref[:,1], 
        t=t_raw, 
        w=1./ref[:,2], 
        #bbox=[None, None], 
        k=3)


    #
    # Now also conpute a spline for the SALT spectrum
    #
    print "Fitting spline to data"
    data_wl_min = data[0,1]
    data_wl_max = data[-1,1]
    t_data_points =  int(numpy.ceil((data_wl_max - data_wl_min) / basepoint_spacing)) \
                     if basepoint_spacing > 0 else int(numpy.fabs(basepoint_spacing))
    t_data_raw = numpy.linspace(data_wl_min+3, data_wl_max-3, t_data_points)
    t_data = numpy.empty(shape=(t_data_raw.shape[0]+4))
    t_data[0] = t_raw[0]-2
    t_data[1] = t_raw[0]-1
    t_data[-2] = t_raw[-1]+1
    t_data[-1] = t_raw[-1]+2
    t_data[2:-2] = t_data_raw[:]

    print data[:,1]
    print data[:,2]
    t_data_good = satisfy_schoenberg_whitney(data[:,1], t_data_raw)
    data_spline =  scipy.interpolate.LSQUnivariateSpline(
        x=data[:,1], 
        y=data[:,2], 
        t=t_data_good, 
        w=1./data[:,3], 
        #bbox=[None, None], 
        k=3)

    # 
    # Now compute the flux calibration factor
    #
    flux_ref = ref_spline(data[:,1])
    flux_data = data_spline(data[:,1])
    flux_cal_factor = flux_ref / flux_data

    numpy.savetxt("f_ref", flux_ref)
    numpy.savetxt("f_data", flux_data)

    data_final = numpy.empty(shape=(data.shape[0], data.shape[1]+4))
    data_final[:, :data.shape[1]] = data[:,:]
    data_final[:, data.shape[1]+0] = flux_ref
    data_final[:, data.shape[1]+1] = flux_data
    data_final[:, data.shape[1]+2] = flux_cal_factor
    data_final[:, data.shape[1]+3] = data[:,2] * flux_cal_factor

    header = """\
Column 1: running wavelength ID
Column 2: wavelength in A
Column 3: raw flux in ADU/pixel
Column 4: RMS in ADU/pixel
Column 5: spline-smoothed flux of reference spectrum
Column 6: spline-smoothed flux of input spectrum
Column 7: flux calibration factor (col. 5 divided by col. 6)
Column 8: Flux-calibrated output spectrum
"""
    print "saving results to %s" % (outfile)
    numpy.savetxt(outfile+".dat", data_final, header=header)
                          

    fig = pyplot.figure()
    ax = fig.add_subplot(111)

    ax.plot(data_final[:,1], data_final[:,-1], label="SALT")
    ax.plot(ref[:,0], ref[:,1], label="Reference", c='grey')
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("flux")
    ax.set_xlim((data_wl_min, data_wl_max))
    ax.set_ylim((0, 14e-16))
#    ax.legend(loc='best', borderaxespad=0.5, prop={'size':9})
    ax.legend(loc='upper right', borderaxespad=0.5, prop={'size':9})

    fig.tight_layout()
    fig.set_size_inches((9,6))
    fig.savefig(outfile+".png", dpi=100)


    #
    # Save file as FITS file as well
    #
    hdu_list = fits.HDUList([fits.PrimaryHDU(data=data_final)])
    hdu_list.writeto(outfile+".fits", clobber=True)

#!/usr/bin/env python

import os
import sys
from astropy.io import fits
import numpy
import scipy

import pysalt
import logging
from optparse import OptionParser

datadir="/work/salt/sandbox_official/polSALT/polsalt/data/"

# based on http://www.sal.wisc.edu/PFIS/docs/rss-vis/archive/protected/pfis/3170/3170AM0010_Spectrograph_Model_Draft_2.pdf
# and https://github.com/saltastro/SALTsandbox/blob/master/polSALT/polsalt/specpolmap.py

def rotate_detector(ndet=0,
                    dy=0, slope=0,
                    xbin=1,ybin=1):

    # center coordinates of each detector in unbinned pixels
    refx = [1024,3162, 5300][ndet]
    refy = 2048

    # compute rotation angle
    angle = numpy.arcsin(slope)

    sin_angle = slope
    cos_angle = numpy.cos(angle)



def rssmodelwave(#grating,grang,artic,cbin,refimg,
        header, img,
        xbin=1, ybin=1,
        y_center=None, x_center=None,
        debug=False,
):
#   compute wavelengths from model (this can probably be done using pyraf spectrograph model)

    logger = logging.getLogger("RSS-2Dmodel")
    ncols = img.shape[0]
    nrows = img.shape[1]

    # compute X/Y position for each pixel
    y,x = numpy.indices(img.shape)
    # also account for binning
    y *= ybin
    x *= xbin

    if (y_center is None):
        y_center = img.shape[0]/2. * ybin
    if (x_center is None):
        x_center = img.shape[1]/2. * xbin
    logger.info("Using (un-binned) center coordinates of x=%.2f, y=%.2f" % (x_center, y_center))

    #
    #
    # Load spectrograph parameters
    #
    spec=numpy.loadtxt(datadir+"spec.txt",usecols=(1,))
    grating_rotation_home_error = spec[0]

    Grat0,Home0,ArtErr,T2Con,T3Con=spec[0:5]
    FCampoly=spec[5:11]

    grating_names=numpy.loadtxt(datadir+"gratings.txt",dtype=str,usecols=(0,))
    #grname=numpy.loadtxt(datadir+"gratings.txt",dtype=str,usecols=(0,))
    grlmm,grgam0=numpy.loadtxt(datadir+"gratings.txt",usecols=(1,2),unpack=True)


    #
    # Load all necessary information from FITS header
    #
    grating_angle = header['GR-ANGLE'] # alpha_C
    articulation_angle = header['CAMANG'] #GRTILT'] # A_C
    grating_name = header['GRATING']

    logger.debug("grating-angle: %f" % (grating_angle))
    logger.debug("articulation angle: %f" % (articulation_angle))
    logger.debug("grating name: %s" % (grating_name))

    
    # get grating data: lines per mm
    #grnum = numpy.where(grname==grating)[0][0]
    #lmm = grlmm[grnum]
    grnum = numpy.where(grating_names==grating_name)[0][0]
    grating_lines_per_mm = grlmm[grating_name == grating_names][0]
    logger.debug("grating lines/mm: %f" % (grating_lines_per_mm))

    #alpha_r = numpy.radians(grang+Grat0)
    alpha_r = numpy.radians(grating_angle+Grat0)
    #beta0_r = numpy.radians(artic*(1+ArtErr)+Home0)-alpha_r
    beta0_r = numpy.radians(articulation_angle*(1+ArtErr)+Home0)-alpha_r
    gam0_r = numpy.radians(grgam0[grnum])

    logger.debug("alpha-r: %f" % (alpha_r))
    logger.debug("beta_r : %f" % (beta0_r))
    logger.debug("gamma_r: %f" % (gam0_r))

    # compute reference wavelength at center of focal plane
    #lam0 = 1e7*numpy.cos(gam0_r)*(numpy.sin(alpha_r) + numpy.sin(beta0_r))/lmm
    lam0 = 1e7*numpy.cos(gam0_r)*(numpy.sin(alpha_r) + numpy.sin(beta0_r))/grating_lines_per_mm
    logger.debug("reference wavelength: %f" % (lam0))

    # compute camera focal length
    ww = (lam0-4000.)/1000.
    fcam = numpy.polyval(FCampoly,ww)
    logger.debug("camera focal length @ 4000A: %f mm" %(fcam))

    # compute dispersion per pixel
    disp = (1e7*numpy.cos(gam0_r)*numpy.cos(beta0_r)/grating_lines_per_mm) / (fcam/.015)
    #disp = (1e7*numpy.cos(gam0_r)*numpy.cos(beta0_r)/lmm)/(fcam/.015)
    logger.debug("dispersion: %f angstroems/pixel [unbinned]" % (disp))


    # 
    # Iteratively compute a lambda for each pixel, refine the focal length as 
    # fct of lambda, and recompute lambda
    # 
    _x = (x - x_center) * 0.015 #/ 3162.
    _y = (y - y_center) * 0.015 # / 2048.
    logger.debug("min/max X: %f / %f" % (numpy.min(x), numpy.max(x)))
    logger.debug("min/max Y: %f / %f" % (numpy.min(y), numpy.max(y)))

    logger.debug("min/max _X [mm]: %f / %f" % (numpy.min(_x), numpy.max(_x)))
    logger.debug("min/max _Y [mm]: %f / %f" % (numpy.min(_y), numpy.max(_y)))

    alpha = numpy.ones(img.shape) * alpha_r
    for iteration in range(4):
        logger.debug("working on iterative correction for fcam(lambda) - iteration %d" % (iteration+1))
        beta = _x/fcam + beta0_r 
        gamma = _y/fcam + gam0_r
        #print beta.shape, gamma.shape

        # compute lambda (1e7 = angstroem/mm)
        _lambda = 1e7 * numpy.cos(gamma) * (numpy.sin(beta) + numpy.sin(alpha)) / grating_lines_per_mm

        L = (_lambda - 4000.) / 1000.
        fcam = numpy.polyval(FCampoly,L)
        #print "ITER", iteration, fcam.shape

        if (debug):
            fits.PrimaryHDU(data=_lambda).writeto("lambda_%d.fits" % (iteration+1), clobber=True)

        
    return _lambda

        
    # now compute F_cam as function of lambda_0
    # use polynomial fit from ZEMAX camera model (that's step from Ken's docu)
    dfcam = 3.162*disp*numpy.polyval([FCampoly[x]*(5-x) for x in range(5)],ww)


    #T2 = -0.25*(1e7*numpy.cos(gam0_r)*numpy.sin(beta0_r)/lmm)/(fcam/47.43)**2 + T2Con*disp*dfcam
    T2 = -0.25*(1e7*numpy.cos(gam0_r)*numpy.sin(beta0_r)/grating_lines_per_mm)/(fcam/47.43)**2 + T2Con*disp*dfcam
    T3 = (-1./24.)*3162.*disp/(fcam/47.43)**2 + T3Con*disp
    T0 = lam0 + T2 
    T1 = 3162.*disp + 3*T3

    return


    # compute normalized X-position (range [-1,1], X=0 is center of middle chip)
    X = (numpy.array(range(cols))+1-cols/2)*cbin/3162.

    lam_X = T0+T1*X+T2*(2*X**2-1)+T3*(4*X**3-3*X)
    return lam_X


if __name__ == "__main__":
    
    logger = pysalt.mp_logging.setup_logging()

    parser = OptionParser()
    parser.add_option("", "--xbin", dest="xbin",
                      help="binning in x-direction (wavelength)",
                      default=2, type=int)
    parser.add_option("", "--ybin", dest="ybin",
                      help="binning in y-direction (spatial)",
                      default=2, type=int)
    parser.add_option("", "--ycenter", dest="ycenter",
                      help="y center position in un-binned pixels",
                      default=2170, type=float)

    (options, cmdline_args) = parser.parse_args()

    fn = cmdline_args[0]
    hdulist = fits.open(fn)


    binning = pysalt.get_binning(hdulist)
    xbin, ybin = options.xbin, options.ybin
    print "binning", binning, xbin, ybin


    wlmap = rssmodelwave(#grating,grang,artic,cbin,refimg,
        header=hdulist[0].header, 
        img=hdulist['SCI'].data,
        xbin=xbin, ybin=ybin,
        y_center=options.ycenter,
    )
    print wlmap.shape
    fits.PrimaryHDU(data=wlmap).writeto(cmdline_args[1], clobber=True)

    pysalt.mp_logging.shutdown_logging(logger)


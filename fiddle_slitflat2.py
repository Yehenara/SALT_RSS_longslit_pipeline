#!/usr/bin/env python


import sys, numpy, scipy, pyfits
import scipy.ndimage
from fiddle_slitflat import compute_profile
import pickle
import logging

import itertools
def polyfit2dx(x, y, z, order=[3,3], ):
    ncoeffs = (order[0] + 1) * (order[1]+1)
    G = numpy.zeros((x.size, ncoeffs))

    if (x.ndim > 1):
        x=x.ravel()
        y=y.ravel()
        z=z.ravel()

    ij = itertools.product(range(order[0]+1), range(order[1]+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = numpy.linalg.lstsq(G, z)
    return m

def polyval2dx(x, y, m, order=[3,3]):

    #order = int(numpy.sqrt(len(m))) - 1

    shape = x.shape
    if (x.ndim > 1):
        x=x.ravel()
        y=y.ravel()

    ij = itertools.product(range(order[0]+1), range(order[1]+1))
    z = numpy.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j

    return z.reshape(shape)


from optscale import polyfit2d, polyval2d


def create_2d_flatfield_from_sky(wl, img, reuse_profile=None):

    logger = logging.getLogger("Create2dVPHFlat")

    n_wl_chunks = 60
    wl_min = numpy.min(wl)
    wl_max = numpy.max(wl)
    wl_steps = (wl_max - wl_min) / n_wl_chunks
    if (wl_steps > 20): wl_steps = 20

    wl_centers = numpy.linspace(wl_min+0.5*wl_steps, wl_max-0.5*wl_steps, num=n_wl_chunks)

    # print wl_centers, wl_centers.shape

    profiles = numpy.empty((wl.shape[0], wl_centers.shape[0]))
    profiles[:,:] = numpy.NaN

    # prepare an undersampled y grid to keep processing times in check
    # based on this grid we can then interpolate up to the full resolution needed during reduction
    ny = 40
    sparse_y = numpy.linspace(-0.1*wl.shape[0], 1.1*wl.shape[0]-1, num=ny, endpoint=True, dtype=numpy.int)
    profiles_sparse = numpy.empty((sparse_y.shape[0], wl_centers.shape[0]))
    profiles_sparse[:, :] = numpy.NaN

    if (reuse_profile is not None and os.path.isfile(reuse_profile)):
        with open(reuse_profile, "rb") as pf:
            (profiles, profiles_sparse) = pickle.load(pf)

    else:
        for i_wl, cwl in enumerate(wl_centers):
            logger.debug("Extracting profile %d for wl %f +/- %f" % (i_wl+1, cwl, wl_steps))

            prof, poly = compute_profile(
                wl=wl,
                img=img,
                line_wl=cwl,
                line_width=wl_steps,
                n_iter=15,
                polyorder=5)

            if (prof == None):
                logger.debug("No data found for %f +/- %f" % (cwl, wl_steps))
                continue

            #print prof.shape
            profiles[:,i_wl] = prof

            profiles_sparse[:, i_wl] = numpy.polyval(poly, sparse_y)

        with open("profiles_sparse", "wb") as pf:
            pickle.dump((profiles, profiles_sparse), pf)

    pyfits.PrimaryHDU(data=profiles).writeto("fiddle_slitflat.fits", clobber=True)

    # normalize all profiles
    ref_row = wl.shape[0] / 2
    profiles = profiles / profiles[ref_row,:]
    pyfits.PrimaryHDU(data=profiles).writeto("fiddle_slitflatnorm.fits", clobber=True)

    # also normalize the sparse profile grid
    ref_row_sparse = sparse_y.shape[0] / 2
    profiles_sparse = profiles_sparse / profiles_sparse[ref_row_sparse, :]
    pyfits.PrimaryHDU(data=profiles_sparse).writeto("fiddle_slitflatnorm_sparse.fits", clobber=True)


    #
    # Now fit a 2-d polynomial to the sparse grid of scaling factors
    #
    logger.info("Fitting 2-D flat-field profile")
    wl_x2d = wl_centers.reshape((1,-1)).repeat(ny, axis=0)
    y_y2d = sparse_y.reshape((-1,1)).repeat(wl_centers.shape[0], axis=1)
    # print wl_x2d.shape, y_y2d.shape
    # print wl_x2d
    # print y_y2d

    # mask out all pixels with NaNs
    good_data = numpy.isfinite(profiles_sparse)
    # dont_use = (profiles_sparse < 0.3) | (profiles_sparse > 1.5)
    # good_data &= ~dont_use

    fit_steps = [pyfits.PrimaryHDU()]
    fit_residuals = [pyfits.PrimaryHDU()]
    order = [1, 5]
    for final_iteration in range(10):
        logger.debug("iteration %d - %d good data points of %d" % (final_iteration+1,  numpy.sum(good_data), profiles_sparse.size))
        # print profiles_sparse.shape

        # poly2d = polyfit2d(x=wl_x2d[good_data], y=y_y2d[good_data], z=profiles_sparse[good_data], order=5)
        # fit2d = polyval2d(x=wl_x2d, y=y_y2d, m=poly2d)

        interpol = scipy.interpolate.SmoothBivariateSpline(
            x=wl_x2d[good_data],
            y=y_y2d[good_data],
            z=profiles_sparse[good_data],
            w=None,
            bbox=[None, None, None, None],
            kx=5, ky=5)

        #poly2d = polyfit2dx(x=wl_x2d[good_data], y=y_y2d[good_data], z=profiles_sparse[good_data], order=order)
        #fit2d = polyval2dx(x=wl_x2d, y=y_y2d, m=poly2d, order=order)

        fit2d = interpol(x=wl_x2d, y=y_y2d, grid=False)

        # print fit2d.shape
        fit_steps.append(pyfits.ImageHDU(data=fit2d))

        residuals = profiles_sparse - fit2d
        _perc = numpy.nanpercentile(residuals[good_data], [16,84,50])
        _median = _perc[2]
        _sigma = 0.5*(_perc[1] - _perc[0])
        _nsigma = 3
        outlier = (residuals > _nsigma*_sigma) | (residuals < -_nsigma*_sigma)
        good_data[outlier] = False

        fit_residuals.append(pyfits.ImageHDU(data=residuals.copy()))

        logger.debug("Iteration %d: median/sigma = %f / %f" % (final_iteration+1, _median, _sigma))
        residuals[~good_data] = numpy.NaN
        #fit_residuals.append(pyfits.ImageHDU(data=residuals))

    pyfits.HDUList(fit_steps).writeto("fiddle_slitflatnorm_sparsefit.fits", clobber=True)
    pyfits.HDUList(fit_residuals).writeto("fiddle_slitflatnorm_sparseresiduals.fits", clobber=True)
    # pyfits.PrimaryHDU(data=wl_x2d).writeto("fiddle_slitflatnorm_x2d.fits", clobber=True)
    # pyfits.PrimaryHDU(data=y_y2d).writeto("fiddle_slitflatnorm_y2d.fits", clobber=True)

    #
    # Now compute the full resolution 2-d frame from the best-fit polynomial
    #
    logger.info("computing full-resolution scaling frame")
    full_y, _ = numpy.indices(img.shape)
    # fullres2d = polyval2dx(x=wl, y=full_y, m=poly2d, order=order)
    fullres2d = interpol(x=wl, y=full_y, grid=False)
    pyfits.PrimaryHDU(data=fullres2d).writeto("fiddle_slitflatnorm_fullres2d.fits", clobber=True)

    pyfits.PrimaryHDU(data=(img/fullres2d)).writeto("fiddle_slitflatnorm_fullresimg2d.fits", clobber=True)

    return fullres2d, interpol

if __name__ == "__main__":

    fn = sys.argv[1]

    hdu = pyfits.open(fn)

    wl = hdu['WAVELENGTH'].data
    img = hdu['SCI.RAW'].data

    reuse_profile = None
    if (len(sys.argv) > 2 and sys.argv[2] == "reuse"):
        reuse_profile = "profiles_sparse"

    create_2d_flatfield_from_sky(wl, img)
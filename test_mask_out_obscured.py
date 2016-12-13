#!/usr/bin/env python

import pyfits
import os
import sys
import numpy
import scipy
import scipy.ndimage



def find_obscured_regions(img, threshold=0.35):

    #img[~numpy.isfinite(img)] = 0.

    #
    # Apply a rough background subtraction
    #
    filter_img = img.copy()
    filter_img[~numpy.isfinite(filter_img)] = 0.
    w = 25
    i0 = scipy.ndimage.filters.gaussian_filter(numpy.median(filter_img[:w, :], axis=0), sigma=10, mode='reflect') #.reshape((1,-1))
    #i0 = scipy.ndimage.filters.median_filter(numpy.median(img[:w, :], axis=0), size=30, mode='reflect') #.reshape((1,-1))
    i1 = numpy.median(filter_img[-w:, :], axis=0) #.reshape((1,-1))
    i1 = scipy.ndimage.filters.gaussian_filter(i1, sigma=10, mode='reflect')
    numpy.savetxt("i0", i0.ravel())
    numpy.savetxt("i1", i1.ravel())

    #bg = numpy.arange(img.shape[0], dtype=numpy.float).reshape((-1,1)) * 0.5*(i1+i0) / img.shape[0]
    bg = numpy.arange(img.shape[0], dtype=numpy.float).reshape((-1,1)) * (i1-i0) / img.shape[0] + i0
    print img.shape, i0.shape, bg.shape

    img = img - bg
    pyfits.PrimaryHDU(data=bg).writeto("bg.fits", clobber=True)
    pyfits.PrimaryHDU(data=img).writeto("bgsub.fits", clobber=True)

    numpy.savetxt("profile.sum", numpy.sum(img, axis=1))

    profile = numpy.zeros((img.shape[0]))
    profile_noise = numpy.ones((img.shape[0]))*1e6
    profile_y = numpy.zeros((img.shape[1]))
    profile_noise_y = numpy.ones((img.shape[1]))*1e6
    print profile.shape, profile_noise.shape

    # for iter in range(3):
    #
    #     profile_2d = numpy.repeat(profile.reshape((-1,1)), img.shape[1], axis=1)
    #     profile_noise_2d = numpy.repeat(profile_noise.reshape((-1,1)), img.shape[1], axis=1)
    #     # profile_y_2d = numpy.repeat(profile_y.reshape((1,-1)), img.shape[0], axis=0)
    #     # profile_noise_y_2d = numpy.repeat(profile_noise_y.reshape((1,-1)), img.shape[0], axis=0)
    #
    #     print profile_2d.shape, img.shape
    #
    #     bad = (img > profile_2d+3*profile_noise_2d) | (img < profile_2d-3*profile_noise_2d)
    #     # bad_y = (img > profile_y_2d+3*profile_noise_y_2d) | (img < profile_y_2d-3*profile_noise_y_2d)
    #     very_bad = bad #& bad_y
    #
    #     img[very_bad] = numpy.NaN #profile_2d[very_bad]
    #     pyfits.PrimaryHDU(data=img).writeto("clean_%d.fits" % (iter+1), clobber=True)

    profile = numpy.nanmean(img, axis=1)
    profile_noise = numpy.nanstd(img, axis=1)
    print profile.shape

    # numpy.savetxt("profile.%d" % (iter+1), profile)
    # numpy.savetxt("profile_noise.%d" % (iter+1), profile_noise)
    # print "iter:", iter, numpy.median(profile), numpy.std(profile), numpy.mean(profile)

    numpy.savetxt("profile", profile)
    numpy.savetxt("profile_noise", profile_noise)

    # #
    # # Now fit a linear gradient to the very edges
    # #
    # i0 = numpy.median(profile[:w])
    # i1 = numpy.median(profile[-w:])
    # print i0, i1
    # gradient = numpy.arange(img.shape[0], dtype=numpy.float)*(i1-i0)/img.shape[0]+i0
    #
    # profile_corr = profile - gradient
    # numpy.savetxt("profile.corr", profile_corr)
    #
    # good = numpy.isfinite(profile_corr)
    # for iter in range(3):
    #     stats = numpy.percentile(profile_corr[good], [16,50,84])
    #     print stats
    #
    #     _sigma = (stats[2] - stats[0])/2.
    #     bad = (profile_corr < (stats[1]-3*_sigma)) | (profile_corr > (stats[1]+3*_sigma))
    #
    #     profile_corr[bad] = 0.
    #     good[bad] = False
    #
    #     numpy.savetxt("profile.corr.%d" % (iter+1), profile_corr)

    # do some profile filtering
    profile2 = scipy.ndimage.filters.median_filter(profile, size=20)
    numpy.savetxt("profile.clean", profile2)

    filter_profile = numpy.array(profile2)
    for iter in range(3):
        good = numpy.isfinite(filter_profile)
        stats = numpy.percentile(filter_profile[good], [16,50,84])
        print stats, numpy.nanmean(filter_profile)

        _med = stats[1]
        _sigma = 0.5*(stats[2] - stats[0])
        filter_profile[ filter_profile > _med+2*_sigma] = numpy.NaN
        filter_profile[ filter_profile < _med-2*_sigma] = numpy.NaN

        numpy.savetxt("profile.clean.%d" % (iter+1), filter_profile)

        # print "median flux:", stats[1]

    bad_columns = profile2 < threshold*_med

    profile2[bad_columns] = numpy.NaN
    numpy.savetxt("profile.cleaned", profile2)

    img[bad_columns, :] = numpy.NaN
    pyfits.PrimaryHDU(data=img).writeto("bgsub+clean.fits", clobber=True)

    return bad_columns



if __name__ == "__main__":

    fn = sys.argv[1]
    hdulist = pyfits.open(fn)
    img = hdulist['SCI'].data.copy()

    bad_columns = find_obscured_regions(img=img)
    print "Found %d bad columns" % (numpy.sum(bad_columns))
#!/usr/bin/env python

import os, sys, pyfits
import numpy
import scipy, scipy.ndimage, scipy.optimize
import scipy.interpolate
import itertools
import math
import logging
import bottleneck

import pysalt

import prep_science

def scaled_sky(p, skyslice):
    return skyslice * p[0]

def sky_residuals(p, imgslice, skyslice):
    ss = scaled_sky(p, skyslice)
    _,x = numpy.indices(imgslice.shape)
    cont = p[2] #p[1]*x+p[2]
    res = (imgslice - (ss+cont)) #* skyslice
    return res[numpy.isfinite(res)]



def sky_wl_residuals(p, imgslice, skyslice):
    ss = scaled_sky(p, skyslice)
    #_,x = numpy.indices(imgslice.shape)
    cont = p[2] #p[1]*x+p[2]
    res = (imgslice - (ss+cont)) * skyslice
    return res[numpy.isfinite(res)]







def minimize_sky_residuals2_spline(img, sky, wl, bpm, vert_size=5, smooth=3, debug_out=True, dl=-10):

    logger = logging.getLogger("SkyScaling2")

    poly2d, data_raw, pf2, data = minimize_sky_residuals2_spline(
        img=img, sky=sky, wl=wl, bpm=bpm, vert_size=vert_size, 
        smooth=smooth, debug_out=debug_out, dl=dl)
    
    minx = numpy.min(data[:,0])
    maxx = numpy.max(data[:,0])

    miny = numpy.min(data[:,1])
    maxy = numpy.max(data[:,1])

    print minx, maxx, miny, maxy

    bp_x = numpy.linspace(0.01*(maxx-minx)+minx, maxx-0.01*(maxx-minx), 7)
    bp_y = numpy.linspace(0.01*(maxy-miny)+miny, maxy-0.01*(maxy-miny), 7)

    print bp_x
    print bp_y

    data_out = numpy.array(data)

    for iteration in range(3):
        valid = numpy.isfinite(data[:,5])
        spline = scipy.interpolate.LSQBivariateSpline(
            x=data[:,0][valid], 
            y=data[:,1][valid], 
            z=data[:,5][valid], 
            tx=bp_x, 
            ty=bp_y, 
            w=None, 
            bbox=[None, None, None, None], 
            kx=3, ky=3)

        fit = spline(x=data[:,0], y=data[:,1], grid=False)

        diff = data[:,5] - fit

        data_out[:,5] = fit
        numpy.savetxt("optscale.data.splinefit_%d" % (iteration), data_out)
        data_out[:,5] = diff
        numpy.savetxt("optscale.data.splinediff_%d" % (iteration), data_out)

        # find noise level
        for it in range(3):
            _perc = numpy.percentile(diff[numpy.isfinite(diff)], [16,50,84])
            _med = _perc[1]
            _sig = 0.5*(_perc[2] - _perc[0])
            outlier = (diff > _med+3*_sig) | (diff < _med-3*_sig)
            diff[outlier] = numpy.NaN

        data[:,5][outlier] = numpy.NaN

    # Now compute the entire full-field scaling frame
    spline2d = spline(
        x=wl.ravel(),
        y=y.ravel(),
        grid=False
    )
    return poly2d, data_raw, pf2, data, spline2d

def minimize_sky_residuals2(img, sky, wl, bpm, vert_size=5, smooth=3, debug_out=True, dl=-10):

    logger = logging.getLogger("SkyScaling2")

    #
    # Do median filtering in wavelength direction to isolate lines from continuum
    #
    sky_lines, sky_continuum= prep_science.filter_isolate_skylines(sky)

    # find block size in wavelength and spatial direction
    print type(wl)
    print wl
    valid_wl = numpy.isfinite(wl)
    if (numpy.sum(valid_wl) <= 1):
        logger.error("Something went wrong, no valid WL data")
        return None

    min_wl = numpy.min(wl[valid_wl]) #bottleneck.nanmin(wl)
    max_wl = numpy.max(wl[valid_wl]) #bottleneck.nanmax(wl)
    if (dl < 0):
        dl = (max_wl-min_wl)/numpy.fabs(dl)
    if (vert_size<0):
        vert_size = img.shape[0]/numpy.fabs(vert_size)

    logger.info("Using blocks of %d pixels and %d angstroems" % (
        vert_size, int(dl)))

    img = numpy.array(img)
    img[bpm > 0] = numpy.NaN

    n_wl_blocks = int(math.ceil((max_wl - min_wl) / dl))
    n_spatial_blocks = int(math.ceil(img.shape[0]/vert_size))
    logger.info("Using %d spatial and %d wavelength blocks" % (n_spatial_blocks, n_wl_blocks))

    print "OptScaling2"+"\n"*10

    data = []
    scaling = numpy.zeros((n_wl_blocks, n_spatial_blocks,5))
    for i_wl, i_spatial in itertools.product(range(n_wl_blocks), range(n_spatial_blocks)):
        
        #
        # Cut out strips in spatial direction
        #
        y_min = i_spatial * vert_size
        y_max = numpy.min([y_min+vert_size, img.shape[0]])
        
        strip_img = img[y_min:y_max]
        strip_sky = sky[y_min:y_max]
        strip_wl = wl[y_min:y_max]
        strip_sky_lines = sky_lines[y_min:y_max]
        strip_sky_continuum = sky_continuum[y_min:y_max]

        #
        # Now select wavelength interval
        #
        wl_min = i_wl*dl + min_wl
        wl_max = wl_min + dl
        in_wl_range = (strip_wl >= wl_min) & (strip_wl <= wl_max)

        sel_img = strip_img[in_wl_range]
        sel_sky = strip_sky[in_wl_range]
        sel_lines = strip_sky_lines[in_wl_range]
        sel_continuum = strip_sky_continuum[in_wl_range]

        p_init = [1.0, 0.0, numpy.median(sel_continuum)]

        try:
            fit_args = (sel_img, sel_lines)
            _fit = scipy.optimize.leastsq(
                sky_wl_residuals,
                p_init, 
                args=fit_args,
                maxfev=500,
                full_output=1)
            best_fit = _fit[0]
        except:
            best_fit = [numpy.NaN]*3

        simple_median = bottleneck.nanmedian(sel_img/sel_sky)
        simple_mean = bottleneck.nanmean(sel_img/sel_sky)
        weight_mean = bottleneck.nansum(sel_img) / bottleneck.nansum(sel_sky)
        # weighted mean = sum(img/sky * sky)/sum(sky) where sky=weight
        data.append([i_wl, i_spatial, simple_mean, simple_median, weight_mean, best_fit[0], best_fit[2]])

        scaling[i_wl, i_spatial,:] = [simple_mean, simple_median, weight_mean, best_fit[0], best_fit[2]]

    data = numpy.array(data)
    data2 = numpy.array(data)
    data2[:,0] = (data[:,0] + 0.5)*dl + min_wl
    data2[:,1] = (data[:,1] + 0.5)*vert_size
    numpy.savetxt("optscale.data", data)
    numpy.savetxt("optscale.data2", data2)

    pyfits.PrimaryHDU(data=scaling).writeto("optscale.data.fits", clobber=True)

    #
    # Do a low-order polynomial fit
    #
    print data2.shape
    data_scaling = data2[:,5]
    data_wl = data2[:,0]
    data_y = data2[:,1]
    for iteration in range(5):
        # y,x = numpy.indices(scaling.shape, dtype=numpy.float)
        good = numpy.isfinite(data_scaling)
        pf2 = polyfit2d(
            x=data_wl[good],
            y=data_y[good],
            z=data_scaling[good],
            order=3)

        fit = polyval2d(x=data_wl, y=data_y, m=pf2)
        #pyfits.PrimaryHDU(data=fit).writeto("optscale.xxx.fits", clobber=True)

        diff = (data_scaling - fit)
        _perc = numpy.percentile(diff[good], [16,50,84])
        _median = _perc[1]
        _sigma = 0.5*(_perc[2] - _perc[0])
        outlier = (diff > _median+3*_sigma) | (diff < _median-3*_sigma)
        data_scaling[outlier] = numpy.NaN

        combined = numpy.empty((data_scaling.shape[0], 4))
        combined[:,0] = data_wl
        combined[:,1] = data_y
        combined[:,2] = fit
        combined[:,3] = diff
        numpy.savetxt("fit_%d" % (iteration+1), combined) #.reshape((-1,combined.shape[2])))

    #
    # Use the 2-d fit to compute a full-resolution scaling image
    #
    y,_ = numpy.indices(img.shape)
    
    fullscale = polyval2d(x=wl, y=y, m=pf2)

    return fullscale, data, pf2, data2

    # #
    # # Now do some 2-d filtering and interpolating
    # #
    # filtered = numpy.zeros(scaling.shape)
    # for plane in range(3):
    #     padded = numpy.zeros((scaling.shape[0]+2*smooth, scaling.shape[1]+2*smooth))
    #     padded[:,:] = numpy.NaN
    #     padded[smooth:-smooth, smooth:-smooth] = scaling[:,:,plane]
        
        
    #     for y,x in itertools.product(range(scaling.shape[0]), range(scaling.shape[1])):
    #         filtered[y,x,plane] = bottleneck.nanmedian(padded[y:y+2*smooth+1, x:x+2*smooth+1])

    #     pyfits.HDUList([
    #         pyfits.PrimaryHDU(),
    #         pyfits.ImageHDU(data=scaling[:,:,plane].T,name="IN"),
    #         pyfits.ImageHDU(data=filtered[:,:,plane].T,name="out"),
    #         ]).writeto("scaling_%d.fits" % (plane), clobber=True)
        
    
    # interpol = scipy.interpolate.RectBivariateSpline(
    #     x=numpy.arange(n_wl_blocks)*dl+min_wl,
    #     y=numpy.arange(n_spatial_blocks)*vert_size,
    #     z=filtered[:,:,1],
    #     s=0
    #     #kind='linear',
    #     #bounds_error=False,
    #     #fill_value=numpy.NaN,
    #     )
    
    # y,_ = numpy.indices(img.shape)
    # full2d = interpol(wl.ravel(), y.ravel(), grid=False).reshape(img.shape)
    # pyfits.PrimaryHDU(data=full2d).writeto("scale2d.fits", clobber=True)

    # return data, filtered, full2d


def minimize_sky_residuals(img, sky, vert_size=5, smooth=20, debug_out=True):

    print img.shape, sky.shape

    n_slices = int(math.ceil(img.shape[0] / float(vert_size)))
    print n_slices
    
    scaling_data = numpy.zeros((n_slices,5))

    for curslice in range(n_slices):
        
        y0 = curslice * vert_size
        y1 = y0+vert_size if ( y0+vert_size < img.shape[0]) else img.shape[0]
        print y0,y1

        img_slice = img[y0:y1, :]
        sky_slice = sky[y0:y1, :]

        p_init = [1.0, 0.0, 0.0]

        fit_args = (img_slice, sky_slice)
        _fit = scipy.optimize.leastsq(
            sky_residuals,
            p_init, 
            args=fit_args,
            maxfev=500,
            full_output=1)
        #print _fit[0]

        # img_slice -= scaled_sky(_fit[0],sky_slice)

        scaling_data[curslice, 0] = 0.5*(y0+y1)
        scaling_data[curslice, 1:4] = _fit[0]


    #
    # Now fit a low-order spline to the scaling profile
    #
    medfilt = scipy.ndimage.filters.median_filter(
        scaling_data[:,1],
        smooth,
        mode='wrap',
        )
    scaling_data[:,-1] = medfilt[:]
    print medfilt

    interp = scipy.interpolate.InterpolatedUnivariateSpline(
        x=scaling_data[:,0],
        y=scaling_data[:,-1],
        k=3,
#        bounds_error=False,
#        fill_value=0,
        )

    full_profile = interp(numpy.arange(img.shape[0]))

    if (debug_out):
        numpy.savetxt("optscale.full", numpy.append(
            numpy.arange(full_profile.shape[0]).reshape((-1,1)),
            full_profile.reshape((-1,1)), axis=1))
        
        numpy.savetxt("optscale.out", scaling_data)

    return scaling_data, full_profile.reshape((-1,1))


def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = numpy.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = numpy.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(numpy.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = numpy.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z


if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()

    if (sys.argv[1] == "v1"):
        img_fn = sys.argv[2]
        img_hdu = pyfits.open(img_fn)

        sky_fn = sys.argv[3]
        sky_hdu = pyfits.open(sky_fn)

        img = img_hdu[0].data

        sky = sky_hdu[0].data

        full_profile = minimize_sky_residuals(img, sky, vert_size=5, smooth=20, debug_out=True)
        skysub = img - (sky * full_profile)
        pyfits.PrimaryHDU(data=img).writeto(sys.argv[3], clobber=True)

    elif (sys.argv[1] == 'v2'):
        fn = sys.argv[2]
        hdulist = pyfits.open(fn)
        
        cosmics = hdulist['COSMICS'].data
        #pyfits.PrimaryHDU(data=cosmics).writeto("cosmics.fits", clobber=True)

        xxx = minimize_sky_residuals2(
            img=hdulist['SCI.RAW'].data.astype(numpy.float)-cosmics, 
            sky=hdulist['SKY.RAW'].data.astype(numpy.float), 
            wl=hdulist['WL_XXX'].data.astype(numpy.float), 
            bpm=hdulist['BPM'].data.astype(numpy.float), 
            vert_size=-95, 
            smooth=3, 
            debug_out=True, 
            dl=-75)

    elif (sys.argv[1] == 'v3'):
        
        hdu = pyfits.open("optscale.data.fits")
        data = hdu[0].data
        print data.shape

        scaling = data[:,:,3]
        print scaling.shape

        smooth = 10

        # filtered = numpy.zeros(scaling.shape)
        # filtered[:,:] = scaling
        # for iteration in range(3):
            
        #     padded = numpy.zeros((filtered.shape[0]+2*smooth, filtered.shape[1]+2*smooth))
        #     padded[:,:] = numpy.NaN
        #     padded[smooth:-smooth, smooth:-smooth] = filtered[:,:]

        #     for y,x in itertools.product(range(filtered.shape[0]), range(filtered.shape[1])):

        #         block = padded[y:y+2*smooth+1, x:x+2*smooth+1]
        #         good_block = numpy.isfinite(block)
                
        #         _perc = numpy.percentile(block[good_block], [16,50,84])
        #         _med = _perc[1]
        #         _sigma = 0.5*(_perc[2] - _perc[0])
                
        #         outlier = (block < _med-3*_sigma) | (block > _med+3*_sigma)
        #         block[outlier] = numpy.NaN

        # output = padded[smooth:-smooth, smooth:-smooth]
        # pyfits.PrimaryHDU(data=output).writeto("optscale.xxx.fits", clobber=True)

        for iteration in range(5):
            y,x = numpy.indices(scaling.shape, dtype=numpy.float)
            good = numpy.isfinite(scaling)
            pf2 = polyfit2d(
                x=x[good],
                y=y[good],
                z=scaling[good],
                order=3)
            fit = polyval2d(x=x, y=y, m=pf2)
            pyfits.PrimaryHDU(data=fit).writeto("optscale.xxx.fits", clobber=True)

            diff = (scaling - fit)
            _perc = numpy.percentile(diff[good], [16,50,84])
            _median = _perc[1]
            _sigma = 0.5*(_perc[2] - _perc[0])
            outlier = (diff > _median+3*_sigma) | (diff < _median-3*_sigma)
            scaling[outlier] = numpy.NaN

            combined = numpy.empty((scaling.shape[0], scaling.shape[1], 4))
            combined[:,:,0] = x
            combined[:,:,1] = y
            combined[:,:,2] = fit
            combined[:,:,3] = diff
            numpy.savetxt("fit_%d" % (iteration+1), combined.reshape((-1,combined.shape[2])))


    else:


        data = numpy.loadtxt("optscale.data")
        smooth = 10

        print data.shape

        minx = numpy.min(data[:,0])
        maxx = numpy.max(data[:,0])

        miny = numpy.min(data[:,1])
        maxy = numpy.max(data[:,1])

        print minx, maxx, miny, maxy

        bp_x = numpy.linspace(0.01*(maxx-minx)+minx, maxx-0.01*(maxx-minx), 7)
        bp_y = numpy.linspace(0.01*(maxy-miny)+miny, maxy-0.01*(maxy-miny), 7)

        print bp_x
        print bp_y

        data_out = numpy.array(data)

        for iteration in range(3):
            valid = numpy.isfinite(data[:,5])
            spline = scipy.interpolate.LSQBivariateSpline(
                x=data[:,0][valid], 
                y=data[:,1][valid], 
                z=data[:,5][valid], 
                tx=bp_x, 
                ty=bp_y, 
                w=None, 
                bbox=[None, None, None, None], 
                kx=3, ky=3)

            fit = spline(x=data[:,0], y=data[:,1], grid=False)

            diff = data[:,5] - fit

            data_out[:,5] = fit
            numpy.savetxt("optscale.data.splinefit_%d" % (iteration), data_out)
            data_out[:,5] = diff
            numpy.savetxt("optscale.data.splinediff_%d" % (iteration), data_out)

            # find noise level
            for it in range(3):
                _perc = numpy.percentile(diff[numpy.isfinite(diff)], [16,50,84])
                _med = _perc[1]
                _sig = 0.5*(_perc[2] - _perc[0])
                outlier = (diff > _med+3*_sig) | (diff < _med-3*_sig)
                diff[outlier] = numpy.NaN
                
            data[:,5][outlier] = numpy.NaN

        os._exit(0)


        # filtered = numpy.zeros(scaling.shape)
        # filtered[:,:] = scaling
        # for iteration in range(3):
            
        #     padded = numpy.zeros((filtered.shape[0]+2*smooth, filtered.shape[1]+2*smooth))
        #     padded[:,:] = numpy.NaN
        #     padded[smooth:-smooth, smooth:-smooth] = filtered[:,:]

        #     for y,x in itertools.product(range(filtered.shape[0]), range(filtered.shape[1])):

        #         block = padded[y:y+2*smooth+1, x:x+2*smooth+1]
        #         good_block = numpy.isfinite(block)
                
        #         _perc = numpy.percentile(block[good_block], [16,50,84])
        #         _med = _perc[1]
        #         _sigma = 0.5*(_perc[2] - _perc[0])
                
        #         outlier = (block < _med-3*_sigma) | (block > _med+3*_sigma)
        #         block[outlier] = numpy.NaN

        # output = padded[smooth:-smooth, smooth:-smooth]
        # pyfits.PrimaryHDU(data=output).writeto("optscale.xxx.fits", clobber=True)

        for iteration in range(5):
            y,x = numpy.indices(scaling.shape, dtype=numpy.float)
            good = numpy.isfinite(scaling)
            pf2 = polyfit2d(
                x=x[good],
                y=y[good],
                z=scaling[good],
                order=3)
            fit = polyval2d(x=x, y=y, m=pf2)
            pyfits.PrimaryHDU(data=fit).writeto("optscale.xxx.fits", clobber=True)

            diff = (scaling - fit)
            _perc = numpy.percentile(diff[good], [16,50,84])
            _median = _perc[1]
            _sigma = 0.5*(_perc[2] - _perc[0])
            outlier = (diff > _median+3*_sigma) | (diff < _median-3*_sigma)
            scaling[outlier] = numpy.NaN

            combined = numpy.empty((scaling.shape[0], scaling.shape[1], 4))
            combined[:,:,0] = x
            combined[:,:,1] = y
            combined[:,:,2] = fit
            combined[:,:,3] = diff
            numpy.savetxt("fit_%d" % (iteration+1), combined.reshape((-1,combined.shape[2])))

    pysalt.mp_logging.shutdown_logging(logger_setup)

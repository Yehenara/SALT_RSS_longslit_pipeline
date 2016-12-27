#!/usr/bin/env python


import sys, numpy, scipy, pyfits
import scipy.ndimage
import logging

write_debug = False

def compute_profile(wl, img, line_wl, line_width=5, n_iter=15, polyorder=5, bad_rows=None):

    y,x = numpy.indices(wl.shape)

    #print wl.shape, img.shape

    #pyfits.PrimaryHDU(data=x).writeto("fiddle_x.fits", clobber=True)
    #pyfits.PrimaryHDU(data=y).writeto("fiddle_y.fits", clobber=True)

    logger = logging.getLogger("ComputeProfile")

    if (bad_rows is not None):
        wl = wl[~bad_rows, :]
        img = img[~bad_rows, :]
        y = y[~bad_rows, :]
        x = x[~bad_rows, :]

    part_of_line = (wl >= line_wl-line_width) & (wl <= line_wl+line_width)

    cut_wl = wl[part_of_line]
    cut_y = y[part_of_line]
    cut_img = img[part_of_line]

    if (numpy.isnan(cut_img).any()):
        # there are invalid pixels in this bin - be safe and not use this data at all
        logger.debug("found at least one pixel with NaN - skipping this data set")
        return None, None

    #print cut_wl.shape, cut_y.shape, cut_img.shape

    combined = numpy.empty((cut_wl.shape[0], 4))
    combined[:,0] = cut_wl
    combined[:,1] = x[part_of_line]
    combined[:,2] = cut_y
    combined[:,3] = cut_img

    if (write_debug):
        out_fn = "fiddle_slitflat.comb.%7.2f-%.2f" % (line_wl, line_width)
        numpy.savetxt(out_fn, combined)
        numpy.save(out_fn, combined)
        #print out_fn

    # print "WL shape:", wl.shape

    output = numpy.empty((wl.shape[0],20))
    output[:,:] = numpy.NaN

    # print y
    # print y[:,0]

    for iy in range(wl.shape[0]):
        #if (bad_rows is not None and bad_rows[iy]):
        #    continue

        wl_match = (wl[iy,:] >= line_wl-line_width) & (wl[iy,:] <= line_wl+line_width)

        # find line intensity
        flux = numpy.sum(img[iy, wl_match])
        output[iy,0] = y[iy,0] #iy
        output[iy,1] = flux

    #
    # run a median filter to get rid of individual hot pixels
    #
    logger.debug("median-filtering the line profile to eliminate hot pixels and cosmics")
    fm = scipy.ndimage.median_filter(output[:,1], size=25, mode='constant', cval=0)
    output[:,2] = fm
    _median = numpy.nanmedian(output[:,1])
    # output[:,1][bad_rows] = _median
    # fm = scipy.ndimage.median_filter(output[:,1], size=25, mode='constant', cval=0)
    # output[:,2] = fm

    # print "output shape:", output.shape

    valid = numpy.isfinite(fm)
    fm[~valid] = -1e99 # this is to prevent runtimewarnings about invalid values encountered
    median_intensity = numpy.median(fm[valid])
    valid &= (fm > 0.3*median_intensity)

    # print "valid", valid.shape

    if (write_debug):
        out_fn = "fiddle_slitflat.comb2.%7.2f-%.2f" % (line_wl, line_width)
        # print out_fn
        with open(out_fn, "w") as f:
           numpy.savetxt(f, output)
           print >>f, "\n"*5
        #print out_fn2

    # print output[:,0][valid]
    # print output[:,2][valid]
    # print "fm:", fm.shape

    fit, poly = None, None
    for iteration in range(n_iter):

        try:
            poly = numpy.polyfit(
                x=output[:,0][valid],
                y=output[:,2][valid],
                deg=polyorder,
            )
        except Exception as e:
            print e
            break

        fit = numpy.polyval(poly, output[:,0])
        residual = fm - fit
        _perc = numpy.nanpercentile(residual[valid], [16,84,50])
        _med = _perc[2]
        _sigma = 0.5*(_perc[1] - _perc[0])
        #print _sigma, _perc

        bad = (residual > 3*_sigma) | (residual < -3*_sigma)
        valid[bad] = False

        output[:,iteration+3] = fit

    output[:,n_iter+3] = output[:,n_iter+2] / output[900,n_iter+2]

    if (write_debug):
        out_fn = "fiddle_slitflat.comb2.%7.2f-%.2f" % (line_wl, line_width)
        with open(out_fn, "w") as f:
           numpy.savetxt(f, output)
           print >>f, "\n"*5
        #print out_fn2

    # print "done with fitting one wl profile"
    return fit, poly


if __name__ == "__main__":

    fn = sys.argv[1]

    hdu = pyfits.open(fn)

    wl = hdu['WAVELENGTH'].data
    img = hdu['SCI'].data

    line_wl = float(sys.argv[2])
    line_width = float(sys.argv[3])
    n_iter = 15
    polyorder=5

    compute_profile(
        wl=wl,
        img=img,
        line_wl=line_wl,
        line_width=line_width,
        n_iter=15,
        polyorder=5
    )

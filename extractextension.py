#!/usr/bin/env python

import os
import sys
import pyfits


if __name__ == "__main__":

    extlist = []
    filelist = []

    suffix = sys.argv[-1]

    found_colon = False
    for i,arg in enumerate(sys.argv[1:-1]):

        if (arg == ":"):
            found_colon = True
            continue

        if (not found_colon):
            extlist.append(arg)
        else:
            filelist.append(arg)


    for fn in filelist:

        if (not os.path.isfile(fn)):
            continue

        hdulist = pyfits.open(fn)
        print "Reading %s" % (fn),
        out_hdulist = [hdulist[0]]

        for extname in extlist:
            try:
                ext = hdulist[extname]
                out_hdulist.append(ext)
            except:
                pass

        out_fn = "%s.%s.fits" % (fn[:-5], suffix)
        print "  writing %s" % (out_fn),
        out_hdulist = pyfits.HDUList(out_hdulist)
        out_hdulist.writeto(out_fn, clobber=True)

        print "  done!"

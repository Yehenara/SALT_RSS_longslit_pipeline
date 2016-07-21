#!/usr/bin/env python

import sys
import numpy
import logging

def calculate_local_noise(data, basepoints, select=None, dumpdebug=False, operator=numpy.nanstd):

    logger = logging.getLogger("LocalNoise")
    logger.debug("Local noise at %d places for %d datapoints" % (basepoints.shape[0], data.shape[0]))

    if (select is None):
        select = data[:,0]
    if (data.ndim == 1):
        data = data.reshape((-1,1))

    if (dumpdebug):
        numpy.savez("data_for_local_noise", data, basepoints, select)

    #print "padding"
    padded = numpy.empty((basepoints.shape[0]+2))
    padded[1:-1] = basepoints[:]
    padded[0] = basepoints[0]
    padded[1] = basepoints[1]
    left_edge = 0.5*(padded[:-2] + padded[1:-1])
    right_edge = 0.5*(padded[1:-1] + padded[2:])

    #print "selecting & computing noise"
    noise = numpy.empty((basepoints.shape[0], data.shape[1]))
    print "Shapes noise/basepoints/padded/data:", noise.shape, basepoints.shape, padded.shape, data.shape

    noise[:,:] = numpy.NaN
    for p in range(basepoints.shape[0]):
        nearby = (select >= left_edge[p]) & (select < right_edge[p])
        subset = data[nearby]
        #print subset.shape

        #_noise = numpy.nanstd(subset,axis=0)
        _noise = operator(data[nearby],axis=0)
        #print p, _noise
        noise[p, :] = _noise

    logger.debug("all done: %s" % (str(noise.shape)))
    return noise

if __name__ == "__main__":

    if (sys.argv[1] == "test"):
        data = numpy.random.random((5495808,1))
        basepoints = numpy.linspace(0, 5495808, 600)
        xpos = numpy.arange(5495808)
        ln = calculate_local_noise(data=data, basepoints=basepoints, select=xpos)
        numpy.savetxt(
            "localnoise.test",
            numpy.append(basepoints.reshape((-1,1)),
                         ln, axis=1)
        )
        print ln.shape
    else:
        print "loading data"
        if (sys.argv[1].endswith(".npy")):
            data = numpy.load(sys.argv[1])
        else:
            data = numpy.loadtxt(sys.argv[1])
        basepoints = numpy.loadtxt(sys.argv[2])

        noise = calculate_local_noise(data, basepoints)
        print "saving results"
        numpy.savetxt("localnoise", noise)

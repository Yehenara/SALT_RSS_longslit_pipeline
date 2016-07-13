#!/usr/bin/env python

import os, sys, pickle, scipy.interpolate, numpy

def write_pickle(fn, **args):
    #print args
    with open(fn, "wb") as pf:
        pickle.dump(args, pf)
    print("pickle complete")

def read_pickle(fn):

    with open(fn, "rb") as pf:
        args = pickle.load(pf)

    print("done unpickling")

    fct = args['fct']
    fct_params = args


    del fct_params['fct']

    #print fct_params
    #print fct

    #print fct_params['bbox']
    #print fct_params['bbox'][1]
    #del fct_params['bbox']

    x = fct(**fct_params)
    print("done re-creatng spline")
    #print x

    return x

if __name__ == "__main__":

    x = numpy.arange(100)
    y = numpy.sin(x) + (numpy.random.rand(x.shape[0])-0.5)*0.1

    k = numpy.arange(5,95,10)

    _min, _max = x[0], x[-1]
    spline_iter = scipy.interpolate.LSQUnivariateSpline(
        x=x,  # allskies[:,0],#[good_point],
        y=y,  # allskies[:,1],#[good_point],
        t=x[5:-5][::5],  # k_wl,
        w=None,  # no weights (for now)
        bbox=[_min, _max],
        k=3,  # use a cubic spline fit
    )

    write_pickle("xxx.test",
                 fct=scipy.interpolate.LSQUnivariateSpline, x=x, y=y, t=x[5:-5][::5], w=None, bbox=[_min,_max], k=3)

    unspline = read_pickle("xxx.test")

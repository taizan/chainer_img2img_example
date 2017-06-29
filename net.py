#!/usr/bin/env python

import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import reporter

from chainer import function
from chainer.utils import type_check


class AutoENC(chainer.Chain):
    def __init__(self):
        super(AutoENC, self).__init__(
            c0=L.Convolution2D(None, 32, 3, 1, 1),
            c1=L.Convolution2D(32, 64, 4, 2, 1),
            c2=L.Convolution2D(64, 128, 4, 2, 1),
            c3=L.Convolution2D(128, 256, 4, 2, 1),
            c4=L.Convolution2D(256, 512, 4, 2, 1),
            c5=L.Convolution2D(512, 512, 3, 1, 1),

            dc5=L.Deconvolution2D(512, 512, 4, 2, 1),
            dc4=L.Deconvolution2D(512, 256, 4, 2, 1),
            dc3=L.Deconvolution2D(256, 128, 4, 2, 1),
            dc2=L.Deconvolution2D(128, 64, 4, 2, 1),
            dc1=L.Convolution2D(64, 32, 3, 1, 1),
            dc0=L.Convolution2D(32, 3, 3, 1, 1),

            bnc0=L.BatchNormalization(32),
            bnc1=L.BatchNormalization(64),
            bnc2=L.BatchNormalization(128),
            bnc3=L.BatchNormalization(256),
            bnc4=L.BatchNormalization(512),
            bnc5=L.BatchNormalization(512),

            bnd5=L.BatchNormalization(512),
            bnd4=L.BatchNormalization(256),
            bnd3=L.BatchNormalization(128),
            bnd2=L.BatchNormalization(64),
            bnd1=L.BatchNormalization(32)
        )

        
    def __call__(self, x_in ):

        e0 = F.relu( self.bnc0( self.c0(x_in) ) )
        e1 = F.relu( self.bnc1( self.c1(e0) ) )
        e2 = F.relu( self.bnc2( self.c2(e1) ) )
        e3 = F.relu( self.bnc3( self.c3(e2) ) )
        e4 = F.relu( self.bnc4( self.c4(e3) ) )
        e5 = F.relu( self.bnc5( self.c5(e4) ) )

        d5 = F.relu( self.bnd5( self.dc5(e5) ) )
        d4 = F.relu( self.bnd4( self.dc4(d5) ) )
        d3 = F.relu( self.bnd3( self.dc3(d4) ) )
        d2 = F.relu( self.bnd2( self.dc2(d3) ) )
        d1 = F.relu( self.bnd1( self.dc1(d2) ) )
        x_out = self.dc0(d1)

        return x_out

class LossEval(chainer.Chain):

    def __init__(self, model, lossfunc=F.mean_absolute_error ):
        super(LossEval, self).__init__()
        self.lossfunc = lossfunc
        with self.init_scope():
            self.model = model 

    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None

        self.y = self.model(*x)
        self.loss = self.lossfunc(self.y, t)
        reporter.report({'loss': self.loss}, self)
        return self.loss


class DIS(chainer.Chain):

    def __init__(self):
        super(DIS, self).__init__(
            c1=L.Convolution2D(None, 32, 3, 1, 1),
            c2=L.Convolution2D(32, 128, 4, 2, 1),
            c3=L.Convolution2D(128, 256, 4, 2, 1),
            c4=L.Convolution2D(256, 512, 4, 2, 1),
            c5=L.Convolution2D(512, 1, 3, 1, 1),

            bnc1=L.BatchNormalization(32),
            bnc2=L.BatchNormalization(128),
            bnc3=L.BatchNormalization(256),
            bnc4=L.BatchNormalization(512),
        )


    def __call__(self, x ):
        h = F.elu(self.bnc1(self.c1(x)))
        h = F.elu(self.bnc2(self.c2(h)))
        h = F.elu(self.bnc3(self.c3(h)))
        h = F.elu(self.bnc4(self.c4(h)))
        h = self.c5(h)

        return h


#!/usr/bin/env python

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
import six
import os
import cv2

from chainer import cuda, optimizers, serializers, Variable
from chainer import training
from chainer.training import extensions

import argparse

import net

from dataset import Image2ImageDataset


def main():
    parser = argparse.ArgumentParser(description='chainer example of img2img learning')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='test',
                        help='Directory to output the test result')
    parser.add_argument('--model', '-m', default='result/model_final',
                        help='model to use')
    parser.add_argument('--filelist', '-fl', default='filelist.dat',
                        help='filelist of dataset')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# batch-size: {}'.format(args.batchsize))
    print('# model: {}'.format(args.model))
    print('')

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            cnn = net.AutoENC()
    
    serializers.load_npz(args.model, cnn)

    if args.gpu >= 0:
        cnn.to_gpu()  # Copy the model to the GPU

    dataset = Image2ImageDataset(args.filelist)
    src , dst  =  dataset.get_example(0)
    x_in = np.zeros((args.batchsize, src.shape[0], src.shape[1], src.shape[2])).astype('f')
    #x_out = np.zeros((batchsize, dst.shape[1], dst.shape[1], dst.shape[2])).astype('f')

    for j in range(args.batchsize):
        src , dst  =  dataset.get_example(j)
        x_in[j,:] =  src
        #x_out[j,:] =  dst
    
    if args.gpu >= 0:
        x_in = cuda.to_gpu(x_in)

    x_out = cnn(x_in)
    output = x_out.data.get()

    for i in range(args.batchsize):
        img = dataset.post_proc(output[i])
        cv2.imwrite( args.out +"/"+ dataset.get_name(i) , img )


if __name__ == '__main__':
    main()

#!/usr/bin/env python

import numpy as np
import chainer
import six
import os

from chainer import cuda, optimizers, serializers, Variable
import cv2


class Image2ImageDataset(chainer.dataset.DatasetMixin):

    def __init__(self, filelist, src_path='./src', dst_path='./dst', dtype=np.float32, train=False, size=(128,128) ):
        if isinstance(filelist, six.string_types):
            with open(filelist) as filelist_file:
                filelist = [f.strip() for f in filelist_file]

        self._filelist = filelist
        self._src_path = src_path
        self._dst_path = dst_path
        self._dtype = dtype
        self._train = train
        self._size = size

    def __len__(self):
        return len(self._filelist)

    def get_name(self, i):
        return self._filelist[i]

    def get_example(self, i):
        #read image from filelist
        path1 = os.path.join(self._src_path, self._filelist[i])
        path2 = os.path.join(self._dst_path, self._filelist[i])
        src_img = cv2.imread(path1, cv2.IMREAD_COLOR)
        dst_img = cv2.imread(path2, cv2.IMREAD_COLOR)

        #resize
        src_img = cv2.resize(src_img,self._size, interpolation = cv2.INTER_AREA ) 
        dst_img = cv2.resize(dst_img,self._size, interpolation = cv2.INTER_AREA ) 

        # add random flip 
        if self._train:
            if np.random.rand() > 0.5:
                src_img = cv2.flip(src_img, 1)
                dst_img = cv2.flip(dst_img, 1)
            if np.random.rand() > 0.8:
                src_img = cv2.flip(src_img, 0)
                dst_img = cv2.flip(dst_img, 0)
        
        #uint to float 
        src_img = np.asarray(src_img, self._dtype)
        dst_img = np.asarray(dst_img, self._dtype)

        #normalize
        src_img = src_img/128.0 -1.0 
        dst_img = dst_img/128.0 -1.0 

        # add random noise
        if self._train:
            noise = np.random.normal(
                0, 0.1 * np.random.rand(), src_img.shape).astype(self._dtype)
            noise += np.random.normal(0, 0.2)
            src_img += noise

        #add channel if image is grayscale
        if src_img.ndim == 2:
            src_img = src_img[:, :, np.newaxis]
        if dst_img.ndim == 2:
            dst_img = dst_img[:, :, np.newaxis]

        #transpose
        src = src_img.transpose(2, 0, 1)
        dst = dst_img.transpose(2, 0, 1)

        return src, dst


#!/usr/bin/env python

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
import six
import os

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
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--snapshot_interval', type=int, default=10000,
                        help='Interval of snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')


    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current

    cnn = net.AutoENC()
    model = net.LossEval(cnn)
    #serializers.load_npz("result/model_iter_xxx", cnn)i

    dataset = Image2ImageDataset("filelist.dat", train=True)

    train_iter = chainer.iterators.SerialIterator( dataset , args.batchsize)

    if args.gpu >= 0:
        cnn.to_gpu()  # Copy the model to the GPU
        model.to_gpu()  # Copy the model to the GPU

    # Setup optimizer parameters.
    opt = optimizers.Adam(alpha=0.00001)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_cnn')
   
    # Set up a trainer
    updater = training.StandardUpdater(train_iter, opt , device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration') 
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        cnn, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        opt, 'optimizer_'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport( trigger=(20, 'iteration'), ))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss']))
    trainer.extend(extensions.ProgressBar(update_interval=20))

    trainer.run()

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Save the trained model
    chainer.serializers.save_npz(os.path.join(args.out, 'model_final'), cnn)
    chainer.serializers.save_npz(os.path.join(args.out, 'optimizer_final'), opt)


if __name__ == '__main__':
    main()

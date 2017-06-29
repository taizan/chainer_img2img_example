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

chainer.cuda.set_max_workspace_size(1024*1024*1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"


def main():
    parser = argparse.ArgumentParser(description='chainer example of img2img learning')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
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
    #serializers.load_npz("result/model_iter_xxx", cnn)

    dis = net.DIS()
    #serializers.load_npz("result/model_dis_iter_xxx", dis)


    dataset = Image2ImageDataset("filelist.dat", train=True)

    train_iter = chainer.iterators.SerialIterator( dataset , args.batchsize)

    if args.gpu >= 0:
        cnn.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()  # Copy the model to the GPU
        #l.to_gpu()

    # Setup optimizer parameters.
    opt = optimizers.Adam(alpha=0.00001)
    opt.setup(cnn)
    opt.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_cnn')
   
    opt_d = chainer.optimizers.Adam(alpha=0.00001)
    opt_d.setup(dis)
    opt_d.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_dec')


    # Set up a trainer
    updater = ganUpdater(
        models=(cnn, dis),
        iterator={
            'main': train_iter,
            #'test': test_iter
             },
        optimizer={
            'cnn': opt,  
            'dis': opt_d},
        device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration') 
    trainer.extend(extensions.dump_graph('cnn/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        cnn, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'model_dis_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        opt, 'optimizer_'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport( trigger=(20, 'iteration'), ))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'cnn/loss', 'cnn/loss_rec','cnn/loss_adv','cnn/loss_l','dis/loss']))
    trainer.extend(extensions.ProgressBar(update_interval=20))

    trainer.run()

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Save the trained model
    chainer.serializers.save_npz(os.path.join(args.out, 'model_final'), cnn)
    chainer.serializers.save_npz(os.path.join(args.out, 'optimizer_final'), opt)



class ganUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        #self.cnn, self.dis, self.l = kwargs.pop('models')
        self.cnn, self.dis = kwargs.pop('models')
        super(ganUpdater, self).__init__(*args, **kwargs)


    def loss_cnn(self, cnn, x_out, dst, dis_out, lam1=100, lam2=1):
        loss_rec = lam1 * ( F.mean_absolute_error(x_out, dst) )
        batchsize,_,w,h = dis_out.data.shape
        loss_adv = lam2 * F.sum( F.softplus(-dis_out) ) / batchsize / w / h
        
        loss = loss_rec + loss_adv 
        chainer.report({'loss': loss,"loss_rec":loss_rec, 'loss_adv': loss_adv }, cnn)        
        
        return loss
        
    def loss_dis(self, dis, dis_real, dis_fake):
        batchsize,_,w,h = dis_real.data.shape
        
        L1 = (2+np.random.rand()) * F.sum(F.softplus(-dis_real)) / batchsize / w / h
        L2 = (2+np.random.rand()) * F.sum(F.softplus(dis_fake)) / batchsize / w / h
        loss =  L1 + L2 
        chainer.report({'loss': loss}, dis)
        return loss


    def update_core(self):        
        xp = self.cnn.xp
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x_in = []
        dst = []
        for i in range(batchsize):
            x_in.append(batch[i][0])
            dst.append(batch[i][1])
        x_in = Variable(xp.asarray(x_in) )
        dst = Variable(xp.asarray(dst) )
       
        # update cnn
        x_out= self.cnn( x_in )
        dis_out = self.dis( x_out )
        cnn_optimizer = self.get_optimizer('cnn')
        cnn_optimizer.update(self.loss_cnn, self.cnn, x_out, dst, dis_out )


        # update dis
        x_out.unchain_backward()
        dis_fake = self.dis( x_out )
        dis_real = self.dis( dst )
        dis_optimizer = self.get_optimizer('dis')
        dis_optimizer.update(self.loss_dis, self.dis, dis_real, dis_fake)

if __name__ == '__main__':
    main()



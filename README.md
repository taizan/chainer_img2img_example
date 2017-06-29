# chainer_img2img_example
an example of chainer learning code

usage:
     train.py -g 0 -b 4 -e 1000
     train_gan.py -g 0 -b 4 -e 1000
 
-g : GPU No. to use
-b : batch size for 1 iteration
-e : epoch number to run

dataset.py : define dataset for src img to dst img.
net.py : define network of cnn
train.py : train code for basic loss
train_gan.py : train code for gan loss

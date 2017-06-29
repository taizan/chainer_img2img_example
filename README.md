# chainer_img2img_example
an example of chainer learning code

usage:

    train.py -g 0 -b 4 -e 1000
    train_gan.py -g 0 -b 4 -e 1000
 
-g : GPU No. to use <br>
-b : batch size for 1 iteration <br>
-e : epoch number to run <br>

dataset.py : define dataset for src img to dst img. <br>
net.py : define network of cnn. <br>
train.py : train code for basic loss.  <br>
train_gan.py : train code for gan loss. <br>

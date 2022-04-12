from dataloader import get_lsun_dataloader
from download_data import get_file
from VAE_train import train
from VAE_imsave import img_main
import argparse
import zipfile
import os

from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--download_data', action='store_true', help='download data from scratch')
opt = parser.parse_args()
print(opt)

if opt.niter <= 10:
    print("Please choose number of iteration greater than 10")
    exit()

if opt.download_data:
    if opt.dataroot is None:
        raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)

    path = get_file("sheep", 'http://dl.yf.io/lsun/objects/sheep.zip')
    with zipfile.ZipFile(file=opt.dataroot + '/sheep.zip') as myzip:
        myzip.extractall(opt.dataroot)
    train_loader, test_loader = get_lsun_dataloader(opt.dataroot + '/sheep')
    train( opt.dataroot+'/sheep',opt.niter)
    img_main(opt.dataroot + '/sheep', opt.niter)

else:
    train_loader, test_loader = get_lsun_dataloader(opt.dataroot + '/sheep')

    train(opt.dataroot+'/sheep',opt.niter)
    img_main(opt.dataroot + '/sheep', opt.niter)


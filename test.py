import sys, os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.loss import *
from ptsemseg.augmentations import *
from ptsemseg.loader.mpiblur_loader import MPIBlurLoader

from scipy import misc


def test(args):
    # Setup Augmentations (must do crop of power of 2, to ensure feature maps from encoder and decoder match each other)
    data_aug = Compose([RandomHorizontallyFlip(), RandomCrop(1248)]) ## use 2496 for SONY

    # Setup Dataloader
    v_loader = MPIBlurLoader(is_transform=True, split='validation', img_size=(args.img_rows, args.img_cols), augmentations=data_aug, img_norm=True)

    valloader = data.DataLoader(v_loader, batch_size=1, num_workers=1)


    # Setup Model
    from unet_1zb_pix2pix import weights_init, _netG
    netG = _netG(input_nc=3, target_nc=1, ngf=64)
    G_solver = torch.optim.Adam(netG.parameters())


#    TODO during training: weight decay, epoch number is incorrect after loading old model, first few epochs result is bad

    # Load model from file
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            netG.load_state_dict(checkpoint['model_state'])
            G_solver.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    # model.eval()

    # Dump test result to file
    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
        # images_val = Variable(images_val.cuda(), volatile=True)
        # labels_val = Variable(labels_val.cuda(), volatile=True)
        G_fake = netG(images_val)
        tmp = G_fake.data.cpu().numpy()
        validation_output_img = tmp[0, 0, :, :]
        ffname = r'c:\tmp\output_' + str(i_val) + '.png'
        misc.imsave(ffname, validation_output_img)

        tmp = images_val.data.cpu().numpy()
        validation_input_img = tmp[0, 0, :, :]
        ffname = r'c:\tmp\input_' + str(i_val) + '.png'
        misc.imsave(ffname, validation_input_img)

        tmp = labels_val.data.cpu().numpy()
        sample_target_img = tmp[0, 0, :, :]
        validation_target_img = sample_target_img.astype(np.uint8)
        ffname = r'c:\tmp\target_' + str(i_val) + '.png'
        misc.imsave(ffname, validation_target_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--img_rows', nargs='?', type=int, default=-1,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=-1,
                        help='Width of the input image')

    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')

    args = parser.parse_args()
    print("in test.py")
    test(args)

import sys, os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import argparse
from tqdm import tqdm
from scipy import misc
from ptsemseg.loader.mpiblur_loader import MPIBlurLoader
from ptsemseg.augmentations import *
#from ptsemseg.loss import *


def train(args):

    # Setup Augmentations
    data_aug = Compose([RandomHorizontallyFlip(), RandomCrop(224)])

    # Setup Dataloader
    t_loader = MPIBlurLoader(is_transform=True, split='training',   img_size=(args.img_rows, args.img_cols), augmentations=data_aug, img_norm=True)
    v_loader = MPIBlurLoader(is_transform=True, split='validation', img_size=(args.img_rows, args.img_cols), img_norm=True)

    n_classes = t_loader.n_classes
    trainloader = torch.utils.data.DataLoader(t_loader, batch_size=4, num_workers=2, shuffle=True)
    valloader = torch.utils.data.DataLoader(v_loader, batch_size=1, num_workers=1)


    # Setup Model
    from unet_1zb_pix2pix import weights_init, _netG
    netG = _netG(input_nc=3, target_nc=1, ngf=64)
    netG.apply(weights_init)
    G_solver = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Load trained model
    #  TODO: weight decay, epoch number is incorrect after loading old model, first few epochs result is bad
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

    for epoch in range(args.n_epoch):

        for i, (images, labels) in enumerate(trainloader):
            #images = Variable(images.cuda())
            #labels = Variable(labels.cuda())
            netG.zero_grad()
            G_fake = netG(images)
            G_loss = F.smooth_l1_loss(G_fake, target=labels)

            # Dump to file (training data)
            if (i + 1) % 3 == 0:
                tmp = G_fake.data.cpu().numpy()
                sample_output_img = tmp[0, 0, :, :]
                misc.imsave(r'c:\tmp\output_img.png', sample_output_img)

                tmp = images.data.cpu().numpy()
                sample_input_img = tmp[0, 0, :, :]
                misc.imsave(r'c:\tmp\input_img.png', sample_input_img)

                tmp = labels.data.cpu().numpy()
                sample_target_img = tmp[0, 0, :, :]
                sample_target_img = sample_target_img.astype(np.uint8)
                misc.imsave(r'c:\tmp\target_img.png', sample_target_img)

            G_loss.backward()
            G_solver.step()

            if (i+1) % 5 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch + 1, args.n_epoch, G_loss.data[0]))

        # Save model to file
        state = {'epoch': epoch + 1,
                 'model_state': netG.state_dict(),
                 'optimizer_state': G_solver.state_dict(), }
        torch.save(state, 'model_latest.pth')

        # Dump to file (validation data)
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            #images_val = Variable(images_val.cuda(), volatile=True)
            #labels_val = Variable(labels_val.cuda(), volatile=True)
            G_fake = netG(images_val)
            tmp = G_fake.data.cpu().numpy()
            validation_output_img = tmp[0, 0, :, :]
            ffname = r'c:\tmp\validation_output_'+ str(i_val)+ '.png'
            misc.imsave(ffname, validation_output_img)

            tmp = images_val.data.cpu().numpy()
            validation_input_img = tmp[0, 0, :, :]
            ffname = r'c:\tmp\validation_input_' + str(i_val) + '.png'
            misc.imsave(ffname, validation_input_img)

            tmp = labels_val.data.cpu().numpy()
            sample_target_img = tmp[0, 0, :, :]
            validation_target_img = sample_target_img.astype(np.uint8)
            ffname = r'c:\tmp\validation_target_' + str(i_val) + '.png'
            misc.imsave(ffname, validation_target_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--img_rows', nargs='?', type=int, default=-1,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=-1,
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=1000,
                        help='# of the epochs')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')

    args = parser.parse_args()
    print("in train.py")
    train(args)

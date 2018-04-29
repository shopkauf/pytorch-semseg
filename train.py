import sys, os
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *
from scipy import misc


def train(args):

    # Setup Augmentations
    #data_aug= Compose([RandomRotate(10),
    #                   RandomHorizontallyFlip()])
    #data_aug= Compose([RandomHorizontallyFlip()])
    data_aug = Compose([RandomHorizontallyFlip(), RandomCrop(224)])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, split='training',   img_size=(args.img_rows, args.img_cols), augmentations=data_aug, img_norm=args.img_norm)
    v_loader = data_loader(data_path, is_transform=True, split='validation', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=2, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=2)


    # Setup Metrics
    #running_metrics = runningScore(n_classes)
        
    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()

        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))

    # Setup Model
    #model = get_model(args.arch, n_classes)
    from unet_1zb_pix2pix import weights_init, _netG
    netG = _netG(input_nc=3, target_nc=1, ngf=64)
    netG.apply(weights_init)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)
    G_solver = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    #loss_fn = cross_entropy2d

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

    best_iou = -100.0 
    for epoch in range(args.n_epoch):

        for i, (images, labels) in enumerate(trainloader):
            #images = Variable(images.cuda())
            #images = Variable(images)
            #labels = Variable(labels.cuda())
            #labels = Variable(labels)

            #optimizer.zero_grad()
            netG.zero_grad()
            #outputs = model(images)
            G_fake = netG(images)

            #loss = loss_fn(input=outputs, target=labels)
            G_loss = F.smooth_l1_loss(G_fake, target=labels)

            if (i + 1) % 10 == 3:
                tmp = G_fake.data.cpu().numpy()
                sample_output_img = tmp[0, 0, :, :]
                misc.imsave(r'c:\tmp\output_img.png', sample_output_img)

                tmp = images.data.cpu().numpy()
                sample_input_img = tmp[0, 0, :, :]
                misc.imsave(r'c:\tmp\input_img.png', sample_input_img)

                tmp = labels.data.cpu().numpy()
                sample_target_img = tmp[0, 0, :, :]
                sample_target_img = sample_target_img.astype(np.uint8);
                misc.imsave(r'c:\tmp\target_img.png', sample_target_img)
            #max_elem = sample_target_img.max()
            #print("max_elem = ", max_elem)

            #loss.backward()
            #optimizer.step()
            G_loss.backward()
            G_solver.step()

            if args.visdom:
                vis.line(
                    X=torch.ones((1, 1)).cpu() * i,
                    Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                    win=loss_window,
                    update='append')

            if (i+1) % 5 == 0:
                #print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]))
                print("Epoch [%d/%d] Loss: %.4f" % (epoch + 1, args.n_epoch, G_loss.data[0]))

        state = {'epoch': epoch + 1,
                 'model_state': netG.state_dict(),
                 'optimizer_state': G_solver.state_dict(), }
        torch.save(state, 'model_latest.pth')

        #model.eval()
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

        #score, class_iou = running_metrics.get_scores()
        #for k, v in score.items():
        #    print(k, v)
        #running_metrics.reset()

        #if score['Mean IoU : \t'] >= best_iou:
        #    best_iou = score['Mean IoU : \t']
        #    state = {'epoch': epoch+1,
        #             'model_state': netG.state_dict(),
        #             'optimizer_state' : G_solver.state_dict(),}
        #    torch.save(state, "{}_{}_best_model.pkl".format(args.arch, args.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=300, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=300, 
                        help='Width of the input image')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=4, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')

    parser.add_argument('--visdom', dest='visdom', action='store_true', 
                        help='Enable visualization(s) on visdom | False by default')
    parser.add_argument('--no-visdom', dest='visdom', action='store_false', 
                        help='Disable visualization(s) on visdom | False by default')
    parser.set_defaults(visdom=False)

    args = parser.parse_args()
    print("David in train.py")
    train(args)

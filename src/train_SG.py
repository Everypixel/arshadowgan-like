# -*- coding: utf-8 -*-

import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp
from piq import ContentLoss

from networks import Discriminator, Generator_with_Refin
from utils import seed_everything, get_training_augmentation, get_validation_augmentation, get_preprocessing, get_image_augmentation
from dataset import ARDataset

import warnings;
warnings.simplefilter('ignore')


def get_parser():
    """A parser for command line arguments """

    parser = argparse.ArgumentParser(description='shadow generation training')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--dataset_path', type=str,
                        help='path to dataset')
    parser.add_argument('--img_size', type=int, default=256,
                        help='input and output image size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument('--lr_G', type=float, default=1e-5,
                        help='learning rate for generator (default: 1e-5)')
    parser.add_argument('--lr_D', type=float, default=1e-5,
                        help='learning rate for discriminator (default: 1e-5)')
    parser.add_argument('--n_epoch', type=int, default=4,
                        help='epochs amount (default: 4)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='amount of workers (default: 4)')
    parser.add_argument('--Gmodel_path', type=str,
                        help='path to saving the generator model')
    parser.add_argument('--Dmodel_path', type=str,
                        help='path to saving the discriminator model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for training (default: cuda:0)')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='encoder for attention part (default: resnet34)')
    parser.add_argument('--betta1', type=float, default=10.0,
                        help='koef for l2_loss (default: 10.0)')
    parser.add_argument('--betta2', type=float, default=1.0,
                        help='koef for per_loss (default: 1.0)')
    parser.add_argument('--betta3', type=float, default=1e-2,
                        help='koef for adv_loss (default: 1e-2)')
    parser.add_argument('--gen_weights', type=str, default='',
                        help='path to generator weights')
    parser.add_argument('--discr_weights', type=str, default='',
                        help='path to discriminator weights')
    parser.add_argument('--patience', type=int, default=5,
                        help='scheduler patience (default: 5)')
    return parser


def train(generator, discriminator, device, n_epoch, optimizer_G, optimizer_D, \
          train_loader, valid_loader, scheduler, losses, models_paths, bettas, writer):
    """A train function for SG training

    Args:
        generator: generator model
        discriminator: discriminator model
        device: torch device for training and validation
        n_epoch: epochs amount
        optimizer_G: optimizer for generator model
        optimizer_D: optimizer for discriminator model
        train_loader: loader for train set
        valid_loader: loader for valid set
        scheduler: scheduler for lr modification
        losses:  list of loss functions
        models_paths: list of model saving paths
        bettas: list of betta values
        writer: tensorboard writer
    """
    # transfer models to device
    generator.to(device)
    discriminator.to(device)

    # variable for validation minimum
    val_common_min = np.inf

    print('Start training!')
    for epoch in range(n_epoch):
        # models in train-mode now
        generator.train()
        discriminator.train()

        # loss results lists
        train_l2_loss = []; train_per_loss = []; train_common_loss = []; train_D_loss = [];
        valid_l2_loss = []; valid_per_loss = []; valid_common_loss = [];

        print('Batch cycle:')
        for batch_i, data in enumerate(tqdm(train_loader)):
            noshadow_image = data[2][:, :3].to(device)
            shadow_image = data[2][:, 3:].to(device)
            robject_mask = torch.unsqueeze(data[3][:, 0], 1).to(device)
            rshadow_mask = torch.unsqueeze(data[3][:, 1], 1).to(device)
            mask = torch.unsqueeze(data[3][:, 2], 1).to(device)

            # prepare model input
            model_input = torch.cat((noshadow_image, mask, robject_mask, rshadow_mask), axis=1)
            # ------------ train generator -------------------------------------
            shadow_mask_tensor1, shadow_mask_tensor2 = generator(model_input)
            result_nn_tensor1 = torch.add(noshadow_image, shadow_mask_tensor1)
            result_nn_tensor2 = torch.add(noshadow_image, shadow_mask_tensor2)

            for_per_shadow_image_tensor = torch.sigmoid(shadow_image)
            for_per_result_nn_tensor1 = torch.sigmoid(result_nn_tensor1)
            for_per_result_nn_tensor2 = torch.sigmoid(result_nn_tensor2)

            # Adversarial ground truths
            valid = Variable(torch.cuda.FloatTensor(np.ones((data[2].size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(torch.cuda.FloatTensor(np.zeros((data[2].size(0), *discriminator.output_shape))), requires_grad=False)

            # compute loss values
            l2_loss = losses[0](shadow_image, result_nn_tensor1) + losses[0](shadow_image, result_nn_tensor2)
            per_loss = losses[1](for_per_shadow_image_tensor, for_per_result_nn_tensor1) + losses[1](for_per_shadow_image_tensor, for_per_result_nn_tensor2)
            gan_loss = losses[2](discriminator(result_nn_tensor2), valid)
            common_loss = bettas[0] * l2_loss + bettas[1] * per_loss + bettas[2] * gan_loss

            optimizer_G.zero_grad()
            common_loss.backward()
            optimizer_G.step()

            # ------------ train discriminator ---------------------------------
            optimizer_D.zero_grad()

            loss_real = losses[2](discriminator(shadow_image), valid)
            loss_fake = losses[2](discriminator(result_nn_tensor2.detach()), fake)
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # ------------------------------------------------------------------
            train_l2_loss.append((bettas[0] * l2_loss).item())
            train_per_loss.append((bettas[1] * per_loss).item())
            train_D_loss.append((bettas[2] * loss_D).item())
            train_common_loss.append(common_loss.item())

        # generator model to eval-mode
        generator.eval()

        # validation
        for batch_i, data in enumerate(valid_loader):
            noshadow_image = data[2][:, :3].to(device)
            shadow_image = data[2][:, 3:].to(device)
            robject_mask = torch.unsqueeze(data[3][:, 0], 1).to(device)
            rshadow_mask = torch.unsqueeze(data[3][:, 1], 1).to(device)
            mask = torch.unsqueeze(data[3][:, 2], 1).to(device)

            # prepare model input
            model_input = torch.cat((noshadow_image, mask, robject_mask, rshadow_mask), axis=1)

            with torch.no_grad():
                shadow_mask_tensor1, shadow_mask_tensor2 = generator(model_input)

            result_nn_tensor1 = torch.add(noshadow_image, shadow_mask_tensor1)
            result_nn_tensor2 = torch.add(noshadow_image, shadow_mask_tensor2)

            for_per_result_shadow_image_tensor = torch.sigmoid(shadow_image)
            for_per_result_nn_tensor1 = torch.sigmoid(result_nn_tensor1)
            for_per_result_nn_tensor2 = torch.sigmoid(result_nn_tensor2)

            # compute loss values
            l2_loss = losses[0](shadow_image, result_nn_tensor1) + losses[0](shadow_image, result_nn_tensor2)
            per_loss = losses[1](for_per_result_shadow_image_tensor, for_per_result_nn_tensor1) + losses[1](for_per_result_shadow_image_tensor, for_per_result_nn_tensor2)
            common_loss = bettas[0] * l2_loss + bettas[1] * per_loss

            valid_per_loss.append((bettas[1] * per_loss).item())
            valid_l2_loss.append((bettas[0] * l2_loss).item())
            valid_common_loss.append(common_loss.item())

        # average loss values
        tr_l2_loss = np.mean(train_l2_loss)
        val_l2_loss = np.mean(valid_l2_loss)
        tr_per_loss = np.mean(train_per_loss)
        val_per_loss = np.mean(valid_per_loss)
        tr_common_loss = np.mean(train_common_loss)
        val_common_loss = np.mean(valid_common_loss)
        tr_D_loss = np.mean(train_D_loss)

        # add results to tensorboard
        writer.add_scalar('tr_l2_loss', tr_l2_loss, epoch)
        writer.add_scalar('val_l2_loss', val_l2_loss, epoch)
        writer.add_scalar('tr_per_loss', tr_per_loss, epoch)
        writer.add_scalar('val_per_loss', val_per_loss, epoch)
        writer.add_scalar('tr_common_loss', tr_common_loss, epoch)
        writer.add_scalar('val_common_loss', val_common_loss, epoch)
        writer.add_scalar('tr_D_loss', tr_D_loss, epoch)

        # print info
        print(f'\nEpoch {epoch}, tr_common loss: {tr_common_loss:.4f}, val_common loss: {val_common_loss:.4f}, D_loss {tr_D_loss:.4f}')

        if val_common_loss <= val_common_min:
            # save the best model
            torch.save(generator.state_dict(), models_paths[0])
            torch.save(discriminator.state_dict(), models_paths[1])
            val_common_min = val_common_loss
            print(f'Model saved!')

        # scheduler step
        scheduler.step(val_common_loss)


def main():
    """Parameters initialization and starting SG model training """
    # read command line arguments
    args = get_parser().parse_args()

    # set random seed
    seed_everything(args.seed)

    # paths to dataset
    train_path = osp.join(args.dataset_path, 'train')
    test_path = osp.join(args.dataset_path, 'test')

    # declare generator and discriminator models
    generator = Generator_with_Refin(args.encoder)
    discriminator = Discriminator(input_shape=(3,args.img_size,args.img_size))

    # load weights
    if args.gen_weights != '':
        generator.load_state_dict(torch.load(args.gen_weights))
        print('Generator weights loaded!')

    if args.discr_weights != '':
        discriminator.load_state_dict(torch.load(args.discr_weights))
        print('Discriminator weights loaded!')

    # declare datasets
    train_dataset = ARDataset(train_path,
                              augmentation=get_training_augmentation(args.img_size),
                              augmentation_images=get_image_augmentation(),
                              preprocessing=get_preprocessing(),)

    valid_dataset = ARDataset(test_path,
                              augmentation=get_validation_augmentation(args.img_size),
                              preprocessing=get_preprocessing(),)

    # declare loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # declare loss functions, optimizers and scheduler
    l2loss = nn.MSELoss()
    perloss = ContentLoss(feature_extractor="vgg16", layers=("relu3_3", ))
    GANloss = nn.MSELoss()

    optimizer_G = torch.optim.Adam([dict(params=generator.parameters(), lr=args.lr_G),])
    optimizer_D = torch.optim.Adam([dict(params=discriminator.parameters(), lr=args.lr_D),])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.9, patience=args.patience)

    # device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # tensorboard
    writer = SummaryWriter()

    # start training
    train(
        generator=generator,
        discriminator=discriminator,
        device=device,
        n_epoch=args.n_epoch,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        train_loader=train_loader,
        valid_loader=valid_loader,
        scheduler=scheduler,
        losses=[l2loss, perloss, GANloss],
        models_paths=[args.Gmodel_path, args.Dmodel_path],
        bettas=[args.betta1, args.betta2, args.betta3],
        writer=writer,
    )


if __name__ == "__main__":
    main()

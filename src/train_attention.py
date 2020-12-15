# -*- coding: utf-8 -*-

import argparse
import os
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp

from utils import seed_everything, get_training_augmentation, get_validation_augmentation, get_preprocessing
from dataset import ARDataset


def get_parser():
    """A parser for command line arguments """

    parser = argparse.ArgumentParser(description='attention training')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--dataset_path', type=str,
                        help='path to dataset')
    parser.add_argument('--img_size', type=int, default=256,
                        help='input and output image size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--n_epoch', type=int, default=4,
                        help='epochs amount (default: 4)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='amount of workers (default: 0)')
    parser.add_argument('--model_path', type=str,
                        help='path to saving the model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for training (default: cuda:0)')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='encoder for attention part (default: resnet34)')
    parser.add_argument('--iou_th', type=float, default=0.5,
                        help='threshold for IoU (default: 0.5)')

    return parser


def train(model, device, n_epoch, optimizer, train_loader, valid_loader, loss,\
          model_path, metric, writer):
    """A train function for attention module training

    Args:
      model: attention model
      device: torch device for training and validation
      n_epoch: epochs amount
      optimizer: optimizer for attention model
      train_loader: loader for train set
      valid_loader: loader for valid set
      loss: loss function for training
      model_path: model saving path
      metric: metric function
      writer: tensorboard writer
    """
    # transfer model to device
    model.to(device)

    max_score = 0
    total_train_steps = len(train_loader)
    total_valid_steps = len(valid_loader)

    print('Start training!')

    for epoch in range(n_epoch):
        # model in train-mode now
        model.train()

        train_loss = 0.0; train_metric = 0.0

        print('Batch cycle:')

        for data in tqdm(train_loader):
            noshadow_image = data[0][:, :3].to(device)
            robject_mask = torch.unsqueeze(data[1][:, 0], 1).to(device)
            rshadow_mask = torch.unsqueeze(data[1][:, 1], 1).to(device)
            mask = torch.unsqueeze(data[1][:, 2], 1).to(device)

            # forward
            model_input = torch.cat((noshadow_image, mask), axis=1)
            model_output = model(model_input)

            # compute train loss and metric
            ground_truth = torch.cat((robject_mask, rshadow_mask), axis=1)
            loss_result = loss(ground_truth, model_output)
            train_metric += metric(ground_truth, model_output).item()

            optimizer.zero_grad()
            loss_result.backward()
            optimizer.step()

            train_loss += loss_result.item()

        # model in eval-mode now
        model.eval()
        valid_loss = 0.0
        valid_metric = 0.0

        # validation
        for data in valid_loader:
            noshadow_image = data[0][:, :3].to(device)
            robject_mask = torch.unsqueeze(data[1][:, 0], 1).to(device)
            rshadow_mask = torch.unsqueeze(data[1][:, 1], 1).to(device)
            mask = torch.unsqueeze(data[1][:, 2], 1).to(device)

            # prepare model input
            model_input = torch.cat((noshadow_image, mask), axis=1)

            with torch.no_grad():
                model_output = model(model_input)

            # compute validation loss and metric
            ground_truth = torch.cat((robject_mask, rshadow_mask), axis=1)
            loss_result = loss(ground_truth, model_output)
            valid_metric += metric(ground_truth, model_output).item()
            valid_loss += loss_result.item()

        train_loss = train_loss / total_train_steps
        train_metric = train_metric / total_train_steps
        valid_loss = valid_loss / total_valid_steps
        valid_metric = valid_metric / total_valid_steps

        # logs to tensorboard
        writer.add_scalar('train_dice_loss', train_loss, epoch)
        writer.add_scalar('train_iou_score', train_metric, epoch)
        writer.add_scalar('valid_dice_loss', valid_loss, epoch)
        writer.add_scalar('valid_iou_score', valid_metric, epoch)

        print(f'\nEpoch {epoch}, train_loss: {train_loss}, train_metric: {train_metric}, valid_loss: {valid_loss}, valid_metric: {valid_metric}')

        # save the best model
        if max_score < valid_metric:
            max_score = valid_metric
            torch.save(model.state_dict(), model_path)
            print('Model saved!')


def main():
    """Parameters initialization and starting attention model training """
    # read command line arguments
    args = get_parser().parse_args()

    # set random seed
    seed_everything(args.seed)

    # paths to dataset
    train_path = osp.join(args.dataset_path, 'train')
    test_path = osp.join(args.dataset_path, 'test')

    # declare Unet model with two ouput classes (occluders and their shadows)
    model = smp.Unet(encoder_name=args.encoder, classes=2, activation='sigmoid',)
    # replace the first convolutional layer in model: 4 channels tensor as model input
    model.encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), \
                                    padding=(3, 3), bias=False)

    # declare datasets
    train_dataset = ARDataset(train_path, augmentation=get_training_augmentation(args.img_size), \
        preprocessing=get_preprocessing(),)

    valid_dataset = ARDataset(test_path, augmentation=get_validation_augmentation(args.img_size), \
        preprocessing=get_preprocessing(),)

    # declare loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, \
                              shuffle=True, num_workers=args.num_workers)

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, \
                              shuffle=False, num_workers=args.num_workers)

    # declare loss function, optimizer and metric
    loss = smp.utils.losses.DiceLoss()
    metric = smp.utils.metrics.IoU(threshold=args.iou_th)
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=args.lr),])

    # tensorboard
    writer = SummaryWriter()

    # device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # start training
    train(
        writer=writer,
        n_epoch=args.n_epoch,
        train_loader=train_loader,
        valid_loader=valid_loader,
        model_path=args.model_path,
        model=model,
        loss=loss,
        metric=metric,
        optimizer=optimizer,
        device=device
    )


if __name__ == "__main__":
    main()

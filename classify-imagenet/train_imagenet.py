"""
Copyright 2020 NVIDIA Corporation.

This file has been modified from a file with the same name in the
DARTS library (licensed under the Apache License, Version 2.0):

https://github.com/quark0/darts

The Apache License for the original version of this file can be
found in this directory. The modifications to this file are subject
to the CC BY-NC-SA 4.0 license located at the root directory.
"""

import argparse
import logging
import os
import sys

from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import to_python_float
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import utils
sys.path.append("..")            # Add fromage location to your PYTHON_PATH.
from fromage import Fromage


parser = argparse.ArgumentParser('imagenet')
# Dataset choices.
parser.add_argument('--data', type=str,
                    default='./imagenet_data/',
                    help='location of the data corpus')
# Optimization choices.
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--optimizer', type=str, default='fromage',
                    choices=['fromage', 'SGD', 'adam'],
                    help='optimizer used for training.')
parser.add_argument('--learning_rate', type=float, default=1e-2,
                    help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.0,
                    help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=90,
                    help='num of training epochs')
parser.add_argument('--grad_clip', type=float,
                    default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float,
                    default=0.1, help='label smoothing')
# Logging choices.
parser.add_argument('--report_freq', type=float, default=100,
                    help='report frequency')
parser.add_argument('--save', type=str, default='EXP',
                    help='experiment name')
parser.add_argument('--results_dir', type=str, default='./results/',
                    help='results directory')
# Misc.
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
# DDP.
parser.add_argument('--local_rank', type=int, default=0,
                    help='rank of process')

args = parser.parse_args()

# Set up DDP.
args.distributed = True
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
args.world_size = torch.distributed.get_world_size()


# Set up logging.
assert args.results_dir
args.save = args.results_dir + '/{}'.format(args.save)
if args.local_rank == 0:
    utils.create_exp_dir(args.save)
logging = utils.Logger(args.local_rank, args.save)
writer = utils.Writer(args.local_rank, args.save)

CLASSES = 1000


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + \
            self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # set seeds
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('args = %s', args)

    # Get data loaders.
    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')

    # data augmentation
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_data = dset.ImageFolder(traindir, transform=train_transform)
    valid_data = dset.ImageFolder(validdir, transform=val_transform)

    # dataset split
    valid_data, test_data = utils.dataset_split(valid_data, len(valid_data))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=8, sampler=train_sampler)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=8)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=8)

    # Create model and loss.
    torch.hub.set_dir('/tmp/hub_cache_%d' % args.local_rank)
    model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=False)
    model = model.cuda()
    model = DDP(model, delay_allreduce=True)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    # Set up network weights optimizer.
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), args.learning_rate, momentum=args.momentum,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'fromage':
        optimizer = Fromage(
            model.parameters(), args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, gamma=0.1, step_size=30)

    # Train.
    global_step = 0
    best_acc_top1 = 0
    for epoch in range(args.epochs):
        # Shuffle the sampler, update lrs.
        train_queue.sampler.set_epoch(epoch + args.seed)

        # Training.
        train_acc_top1, train_acc_top5, train_obj, global_step = train(
            train_queue, model, criterion_smooth, optimizer, global_step)
        logging.info('epoch %d train_acc %f', epoch, train_acc_top1)
        writer.add_scalar('train/loss', train_obj, global_step)
        writer.add_scalar('train/acc_top1', train_acc_top1, global_step)
        writer.add_scalar('train/acc_top5', train_acc_top5, global_step)
        writer.add_scalar('train/lr', optimizer.state_dict()[
            'param_groups'][0]['lr'], global_step)

        # Validation.
        valid_acc_top1, valid_acc_top5, valid_obj = infer(
            valid_queue, model, criterion)
        logging.info('valid_acc_top1 %f', valid_acc_top1)
        logging.info('valid_acc_top5 %f', valid_acc_top5)
        writer.add_scalar('val/acc_top1', valid_acc_top1, global_step)
        writer.add_scalar('val/acc_top5', valid_acc_top5, global_step)
        writer.add_scalar('val/loss', valid_obj, global_step)

        # Test
        test_acc_top1, test_acc_top5, test_obj = infer(
            test_queue, model, criterion)
        logging.info('test_acc_top1 %f', test_acc_top1)
        logging.info('test_acc_top5 %f', test_acc_top5)
        writer.add_scalar('test/acc_top1', test_acc_top1, global_step)
        writer.add_scalar('test/acc_top5', test_acc_top5, global_step)
        writer.add_scalar('test/loss', test_obj, global_step)

        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True

        if args.local_rank == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc_top1': best_acc_top1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.save)

        # Update LR.
        scheduler.step()

    writer.flush()


def train(train_queue, model, criterion, optimizer, global_step):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.train()
    for step, (data, target) in enumerate(train_queue):
        n = data.size(0)
        data = data.cuda()
        target = target.cuda()

        # Forward.
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)

        # Backward and step.
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Calculate the accuracy.
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
        prec1 = utils.reduce_tensor(prec1, args.world_size)
        prec5 = utils.reduce_tensor(prec5, args.world_size)

        objs.update(to_python_float(reduced_loss), n)
        top1.update(to_python_float(prec1), n)
        top5.update(to_python_float(prec5), n)

        if (step + 1) % args.report_freq == 0:
            current_lr = list(optimizer.param_groups)[0]['lr']
            logging.info('train %03d %e %f %f lr: %e', step,
                         objs.avg, top1.avg, top5.avg, current_lr)
        global_step += 1

    return top1.avg, top5.avg, objs.avg, global_step


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for step, (data, target) in enumerate(valid_queue):
            data = data.cuda()
            target = target.cuda()

            logits = model(data)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if (step + 1) % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step,
                             objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()

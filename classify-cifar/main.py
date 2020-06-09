'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import sys
sys.path.append('..')
from fromage import Fromage

from models import *

from tqdm import tqdm
import numpy as np
import pickle


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', type=float, help='initial learning rate')
parser.add_argument('--optim', type=str, help='optimizer, either sgd or fromage')
parser.add_argument('--p_bound', type=float, help='regulariser on model class')
parser.add_argument('--seed', type=int, help='random seed')
parser.add_argument('--bsz', type=int, help='batch size')
args = parser.parse_args()

if args.optim is None: raise Exception("Must supply --optim")
if args.lr is None: raise Exception("Must supply --lr")
if args.seed is None: raise Exception("Must supply --seed")
if args.bsz is None: raise Exception("Must supply --bsz")

writer = SummaryWriter(log_dir='logs/'+f'{args.optim}-{args.lr}-{args.bsz}-p_bound-{args.p_bound}-seed-{args.seed}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 350
milestones = [150, 250] # which epochs to divide learning rate by 10

torch.manual_seed(args.seed)
# cudnn.deterministic = True
# cudnn.benchmark = False
cudnn.benchmark = True
np.random.seed(args.seed)
print(f"Random seed set to {args.seed} for torch and numpy")

# Data
print('\n==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bsz, shuffle=True, num_workers=8)
print(f"\nUsing train batch size {args.bsz}\n")

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('\n==> Building model..')
net = PreActResNet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)

criterion = nn.CrossEntropyLoss()
if args.optim == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0, weight_decay=0.0)
elif args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.optim == 'fromage':
    optimizer = Fromage(net.parameters(), lr=args.lr, p_bound=args.p_bound)
else:
    raise Exception("Unsupported optim")


scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss / len(trainloader), correct/total

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return test_loss / len(testloader), correct/total

for epoch in range(0, epochs):
    for group in optimizer.param_groups: print(f"\nEpoch: {epoch}, lr {group['lr']}")
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    scheduler.step()
    print(  f"train loss {round(train_loss,3)} // "+
            f"train_acc {round(train_acc,3)} // "+
            f"test_loss {round(test_loss,3)} // "+
            f"test_acc {round(test_acc,3)}"     )
    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('train_acc', train_acc, epoch)
    writer.add_scalar('test_acc', test_acc, epoch)

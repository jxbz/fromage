import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import argparse
from tqdm import tqdm
import os
import sys

sys.path.append('..')
from fromage import Fromage
from architecture.generator import Generator
from architecture.discriminator import Discriminator
from fid.fid_score import calculate_fid_given_paths

#########################################
#### Parse arguments ####################
#########################################

parser = argparse.ArgumentParser(description='Class-conditional GAN in Pytorch')
parser.add_argument('--lrG', type=float, default=0.01, help='initial learning rate in G')
parser.add_argument('--lrD', type=float, default=0.01, help='initial learning rate in D')
parser.add_argument('--optim', type=str, default='fromage', help='optimizer, either fromage, sgd or adam')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--bsz', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=120, help='total number of training epochs')
parser.add_argument('--decay', type=int, default=100, help='epoch at which to divide lr by 10')
parser.add_argument('--fid', type=int, default=10, help='epoch interval between FID calculation')
parser.add_argument('--D_per_G', type=int, default=1, help='number of D steps per G step')
args = parser.parse_args()

print("\n==> Arguments passed in were:\n")
for key,value in args.__dict__.items(): print(f"{key}: {value}")
print("")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#########################################
#### Prepare data #######################
#########################################

print("\n==> Downloading CIFAR-10 dataset...\n")

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalizes pixels to be in range (-1,1)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bsz, shuffle=True, num_workers=8)

trainsubset = torch.utils.data.Subset(trainset, range(0,10000)) # for purposes of FID calculation
trainsubloader = torch.utils.data.DataLoader(trainsubset, batch_size=args.bsz, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bsz, shuffle=False, num_workers=8)

if not os.path.exists('samples/train'): os.mkdir('samples/train')
if not os.path.exists('samples/test'): os.mkdir('samples/test')
if not os.path.exists('samples/fake'): os.mkdir('samples/fake')

print("\n==> Saving images for FID calculation...\n")

def save_loader(loader, dir, net=None):
    count = 0
    label_dist = np.zeros(10)
    for it, (data, label) in enumerate(tqdm(loader)):
        if net is None:
            images = data
        else:
            images = net(z=None,label=label.to(device))

        for idx in range(images.size(0)):
            fname = os.path.join('samples', dir, f'image_{count}.png')
            torchvision.utils.save_image(images[idx, :, :, :], fname, normalize=True)
            label_dist[label[idx]]+=1
            count += 1
    return label_dist

def evaluate_fid(paths):
    return calculate_fid_given_paths(paths, batch_size=args.bsz, cuda=(device=='cuda'), dims=2048)

print(f"Saving train images to samples/train.")
label_dist = save_loader(trainsubloader, 'train')
print(f"Label distribution is: {label_dist}\n")

print(f"Saving test images to samples/test.")
label_dist = save_loader(testloader, 'test')
print(f"Label distribution is: {label_dist}\n")

print(f"Evaluating train--test FID for sanity check...")
train_test_fid = evaluate_fid(paths=['samples/train','samples/test'])
print(f"Train--test FID is: {train_test_fid}")

#########################################
#### Build G and D ######################
#########################################

netG = Generator().to(device)
netD = Discriminator().to(device)

if args.optim == 'sgd': 
    optimizer = torch.optim.SGD
    kwargs = {}
elif args.optim == 'fromage': 
    optimizer = Fromage
    kwargs = {}
elif args.optim == 'adam': 
    optimizer = torch.optim.Adam
    kwargs = {'betas': (0.0, 0.999), 'eps': 1e-08}
else: raise Exception("Unsupported optim")

optG = optimizer(netG.parameters(), lr=args.lrG, **kwargs)
optD = optimizer(netD.parameters(), lr=args.lrD, **kwargs)

schedulerG = torch.optim.lr_scheduler.MultiStepLR(optG, milestones=[args.decay], gamma=0.1)
schedulerD = torch.optim.lr_scheduler.MultiStepLR(optD, milestones=[args.decay], gamma=0.1)


#########################################
#### Train ##############################
#########################################

def train():
    print("Training...")

    netG.train()
    netD.train()

    for it, (data, label) in enumerate(tqdm(trainloader)):
        data, label = data.to(device), label.to(device)
        
        # Train D every iteration
        D_fake = netD(netG(z=None,label=label), label)
        D_real = netD(data, label)
        D_loss = torch.clamp(D_fake, min=-1).mean() - torch.clamp(D_real, max=1).mean()

        optD.zero_grad()
        D_loss.backward()
        optD.step()

        # Train G every D_per_G iterations
        if it % args.D_per_G == 0:
            D_fake = netD(netG(z=None,label=label), label)
            G_loss = - D_fake.mean()

            optG.zero_grad()
            G_loss.backward()
            optG.step()

#########################################
#### Evaluate ###########################
#########################################

def test():
    netG.eval()

    print("Evaluating train--fake FID...")
    save_loader(trainsubloader, 'fake', net=netG)
    train_fid = evaluate_fid(paths=['samples/train','samples/fake'])
    print(f"Train--fake FID is: {train_fid}\n")

    print("Evaluating test--fake FID...")
    save_loader(testloader, 'fake', net=netG)
    test_fid = evaluate_fid(paths=['samples/test','samples/fake'])
    print(f"Test--fake FID is: {test_fid}\n")

#########################################
#### Main loop ##########################
#########################################

# Fix a latent z for visualisation during training.
z = torch.randn(10*10, 20*4, device=device)
label = torch.remainder(torch.arange(100, device=device),10)

for epoch in range(0, args.epochs):
    print(f"\n==> Epoch {epoch} \n")

    netG.eval()
    images = netG(z, label)
    torchvision.utils.save_image(images, f'samples/gif/epoch{epoch}.jpg',nrow=10,normalize=True)
    
    if epoch % args.fid == 0:
        test()
    train()
    schedulerG.step()
    schedulerD.step()

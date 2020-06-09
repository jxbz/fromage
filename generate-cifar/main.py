import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import argparse
from tqdm import tqdm
import os
import sys

from fromage import Fromage
from lars import Lars

from architecture.generator import Generator
from architecture.discriminator import Discriminator
from fid.fid_score import calculate_fid_given_paths

#########################################
#### Parse arguments ####################
#########################################

parser = argparse.ArgumentParser(description='Class-conditional GAN in Pytorch')
parser.add_argument('--optim', type=str, default='fromage', help='optimizer')
parser.add_argument('--initial_lr', type=float, default=0.01, help='initial learning rate')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--bsz', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=121, help='total number of training epochs')
parser.add_argument('--fid', type=int, default=10, help='epoch interval between FID calculation')
parser.add_argument('--D_per_G', type=int, default=1, help='number of D steps per G step')
args = parser.parse_args()

print("\n==> Arguments passed in were:\n")
for key,value in args.__dict__.items(): print(f"{key}: {value}")
print("")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_dir = f'logs/paper/{args.optim}-{args.initial_lr}-seed{args.seed}/'
writer = SummaryWriter(log_dir=log_dir)
os.mkdir(log_dir + 'samples')

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

if not os.path.exists(log_dir+'samples/train'): os.mkdir(log_dir+'samples/train')
if not os.path.exists(log_dir+'samples/test'): os.mkdir(log_dir+'samples/test')
if not os.path.exists(log_dir+'samples/fake'): os.mkdir(log_dir+'samples/fake')
if not os.path.exists(log_dir+'samples/gif'): os.mkdir(log_dir+'samples/gif')

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
            fname = os.path.join(log_dir+'samples', dir, f'image_{count}.png')
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
train_test_fid = evaluate_fid(paths=[log_dir+'samples/train',log_dir+'samples/test'])
print(f"Train--test FID is: {train_test_fid}")

#########################################
#### Build G and D ######################
#########################################

netG = Generator().to(device)
netD = Discriminator().to(device)

print("Generator:")
print(f"{sum(p.numel() for p in netG.parameters())} parameters")
print(f"{len(list(netG.parameters()))} tensors")

print("\nDiscriminator:")
print(f"{sum(p.numel() for p in netD.parameters())} parameters")
print(f"{len(list(netD.parameters()))} tensors")

if args.optim == 'sgd': 
    optG = torch.optim.SGD(netG.parameters(), lr=args.initial_lr)
    optD = torch.optim.SGD(netD.parameters(), lr=args.initial_lr)
elif args.optim == 'fromage': 
    optG = Fromage(netG.parameters(), lr=args.initial_lr)
    optD = Fromage(netD.parameters(), lr=args.initial_lr)
elif args.optim == 'lars':
    optG = Lars(netG.parameters(), lr=args.initial_lr)
    optD = Lars(netD.parameters(), lr=args.initial_lr)
elif args.optim == 'adam':
    optG = torch.optim.Adam(netG.parameters(), lr=args.initial_lr, betas=(0.0, 0.999), eps=1e-08)
    optD = torch.optim.Adam(netD.parameters(), lr=args.initial_lr, betas=(0.0, 0.999), eps=1e-08)
else: raise Exception("Unsupported optim")

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
    train_fid = evaluate_fid(paths=[log_dir+'samples/train',log_dir+'samples/fake'])
    print(f"Train--fake FID is: {train_fid}\n")

    print("Evaluating test--fake FID...")
    save_loader(testloader, 'fake', net=netG)
    test_fid = evaluate_fid(paths=[log_dir+'samples/test',log_dir+'samples/fake'])
    print(f"Test--fake FID is: {test_fid}\n")
    return train_fid, test_fid

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
    torchvision.utils.save_image(images, log_dir+f'samples/gif/epoch{epoch}.jpg',nrow=10,normalize=True)

    for name,p in netG.named_parameters():
        writer.add_scalar(f'G_norms/{name}', (p*p).mean().sqrt(), epoch)
    for name,p in netD.named_parameters():
        writer.add_scalar(f'D_norms/{name}', (p*p).mean().sqrt(), epoch)

    if epoch==100:
        print("decaying")
        for group in optG.param_groups: group['lr'] /= 10
        for group in optD.param_groups: group['lr'] /= 10
    
    if epoch % args.fid == 0:
        train_fid, test_fid = test()
        writer.add_scalar('FID/train', train_fid, epoch)
        writer.add_scalar('FID/test', test_fid, epoch)
    train()

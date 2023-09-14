#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
import fastcore.all as fc
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision.transforms.functional as TF
from torch.utils.data import default_collate, DataLoader
import torch.optim as optim
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [2, 2]


dataset_nm = 'mnist'
x,y = 'image', 'label'
ds = load_dataset(dataset_nm)


def transform_ds(b):
    b[x] = [TF.to_tensor(ele) for ele in b[x]]
    return b

dst = ds.with_transform(transform_ds)
plt.imshow(dst['train'][0]['image'].permute(1,2,0));


bs = 1024
class DataLoaders:
    def __init__(self, train_ds, valid_ds, bs, collate_fn, **kwargs):
        self.train = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn, **kwargs)
        self.valid = DataLoader(valid_ds, batch_size=bs*2, shuffle=False, collate_fn=collate_fn, **kwargs)

def collate_fn(b):
    collate = default_collate(b)
    return (collate[x], collate[y])

dls = DataLoaders(dst['train'], dst['test'], bs=bs, collate_fn=collate_fn)
xb,yb = next(iter(dls.train))
xb.shape, yb.shape


class Reshape(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.reshape(self.dim)


def conv(ni, nf, ks=3, s=2, act=nn.ReLU, norm=None):
    layers = [nn.Conv2d(ni, nf, kernel_size=ks, stride=s, padding=ks//2)]
    if norm:
        layers.append(norm)
    if act:
        layers.append(act())
    return nn.Sequential(*layers)

def _conv_block(ni, nf, ks=3, s=2, act=nn.ReLU, norm=None):
    return nn.Sequential(
        conv(ni, nf, ks=ks, s=1, norm=norm, act=act),
        conv(nf, nf, ks=ks, s=s, norm=norm, act=act),
    )

class ResBlock(nn.Module):
    def __init__(self, ni, nf, s=2, ks=3, act=nn.ReLU, norm=None):
        super().__init__()
        self.convs = _conv_block(ni, nf, s=s, ks=ks, act=act, norm=norm)
        self.idconv = fc.noop if ni==nf else conv(ni, nf, ks=1, s=1, act=None)
        self.pool = fc.noop if s==1 else nn.AvgPool2d(2, ceil_mode=True)
        self.act = act()
    
    def forward(self, x):
        return self.act(self.convs(x) + self.idconv(self.pool(x)))


def cnn_classifier():
    return nn.Sequential(
        ResBlock(1, 8, norm=nn.BatchNorm2d(8)),
        ResBlock(8, 16, norm=nn.BatchNorm2d(16)),
        ResBlock(16, 32, norm=nn.BatchNorm2d(32)),
        ResBlock(32, 64, norm=nn.BatchNorm2d(64)),
        ResBlock(64, 64, norm=nn.BatchNorm2d(64)),
        conv(64, 10, act=False),
        nn.Flatten(),
    )


def kaiming_init(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight)        


model = cnn_classifier()
model.apply(kaiming_init)
lr = 0.1
max_lr = 0.3
epochs = 5
opt = optim.AdamW(model.parameters(), lr=lr)
sched = optim.lr_scheduler.OneCycleLR(opt, max_lr, total_steps=len(dls.train), epochs=epochs)
for epoch in range(epochs):
    for train in (True, False):
        accuracy = 0
        dl = dls.train if train else dls.valid
        for xb,yb in dl:
            preds = model(xb)
            loss = F.cross_entropy(preds, yb)
            if train:
                loss.backward()
                opt.step()
                opt.zero_grad()
            with torch.no_grad():
                accuracy += (preds.argmax(1).detach().cpu() == yb).float().mean()
        if train:
            sched.step()
        accuracy /= len(dl)
        print(f"{'train' if train else 'eval'}, epoch:{epoch+1}, loss: {loss.item():.4f}, accuracy: {accuracy:.4f}")


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.utils.data import random_split, DataLoader
from utils.caf_dataloader import GK2A
import numpy as np
from models.caf_SIANet import sianet
from utils import progress_bar
import os

train_dataset = GK2A(data_root='/scratch/q593a18/workspace/PROJECTS/caf/output/gk2a_2020_20len_30min_org_CLD_IR105_WV063.npy', 
                               resize_width=128,
                               is_train=True)
val_dataset = GK2A(data_root='/scratch/q593a18/workspace/PROJECTS/caf/output/gk2a_2021_20len_30min_org_CLD_IR105_WV063.npy', 
                               resize_width=128,
                               is_train=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                          num_workers=16, pin_memory=True, prefetch_factor=2, 
                          persistent_workers=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, 
                        num_workers=16, pin_memory=True, prefetch_factor=2, 
                        persistent_workers=False, drop_last=True)

model = sianet()
model = model.cuda()
model = torch.nn.DataParallel(model)
cudnn.benchmark = True

optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4, weight_decay=0.1)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.long().cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f |' % (train_loss/(batch_idx+1)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.long().cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f |'
                         % (test_loss/(batch_idx+1)))

        state = {
            'net': model.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')

for epoch in range(0, 100):
    train(epoch)
    test(epoch)
    scheduler.step()

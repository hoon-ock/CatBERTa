import numpy as np
import torch, transformers, os
import torch.nn as nn
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from model.classifiers import BinaryClassifier, MultiLabelClassifier
from model.dataset import MyDataset, PretrainDataset
import wandb
from tqdm import tqdm
import datetime


def pretrain_fn(data_loader, model, optimizer, device):
    model.train()
    lr_list = []
    train_losses = []
    print('training...')
    for batch in tqdm(data_loader):
        ids = batch["ids"].to(device, dtype=torch.long)
        masks = batch["masks"].to(device, dtype=torch.long)
        ads_size = batch["ads_size"].to(device, dtype=torch.int)
        ads_class = batch["ads_class"].to(device, dtype=torch.int)
        bulk_class = batch["bulk_class"].to(device, dtype=torch.int)
        optimizer.zero_grad()
        outputs = model(ids, masks) # need to set the shape of output
        # define loss
        loss = nn.CrossEntropyLoss()(outputs[0], ads_class)
        loss += nn.CrossEntropyLoss()(outputs[1], bulk_class)
        loss += nn.CrossEntropyLoss()(outputs[2], ads_size)

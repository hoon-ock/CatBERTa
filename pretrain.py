from _playground.transformer_encoder import Transformer, PositionalEncoding
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import os
import json


class MaterialDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)

        # randomly mask 15% of tokens in the input sequence
        input_ids = encoding#.ids
        mask_indices = torch.rand(len(input_ids)) < 0.15
        masked_indices = mask_indices.nonzero(as_tuple=True)[0]
        input_ids = torch.tensor(input_ids)
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        # create the labels tensor
        # set the label values
        labels = torch.tensor(encoding).clone()
        labels[~mask_indices] = -100  # set non-masked tokens to -100 (ignore index)

        # create the mask tensor to ignore padding tokens
        mask = torch.tensor([1] * len(input_ids), dtype=torch.long)
        mask[masked_indices] = 0
        mask = (mask != self.tokenizer.pad_token_id).view(-1)

        return input_ids, labels, mask


def train(model, optimizer, train_loader, criterion, device):
    model.train()
    total_loss = 0
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, labels, pad_mask = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        pad_mask = pad_mask.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
        # Ignore padding tokens
        pad_mask = (pad_mask != 1).view(-1)
        # print(pad_mask.shape)
        # print(loss.shape)
        #loss[pad_mask] = 0
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


if __name__ == "__main__":
    # Set up hyperparameters
    import datetime
    import glob 

    # Collect text data for training
    text_files = glob.glob("./data/train/random*.txt")
    texts = [open(txt, 'r').read() for txt in text_files]
    
    # Set up tokenizer and data loader  
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="./model/CAT_tokenizer.json", tokenizer_impl=ByteLevelBPETokenizer)
    tokenizer.add_special_tokens({"sep_token": "[SEP]", "cls_token": "[CLS]", "pad_token": "[PAD]", "mask_token": "[MASK]", "unk_token": "[UNK]"})
    print("vocab size:", tokenizer.vocab_size)
    
    # Set up hyperparameters
    vocab_size = tokenizer.vocab_size
    d_model = 256
    hidden_size = 512
    num_layers = 4
    num_heads = 8
    dropout = 0.1
    batch_size = 32
    num_epochs = 100
    lr = 5e-4
    max_length = 20

    dataset = MaterialDataset(texts, tokenizer, max_length)  # Dummy dataset
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up model and optimizer
    model = Transformer(vocab_size, d_model, hidden_size,num_layers, num_heads, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ignore_index = [tokenizer.pad_token_id, tokenizer.mask_token_id]
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    #criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Move model and data to device
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU available. Assigned to device:", device)
    else:
        device = torch.device("cpu")
        print("GPU not available. Using CPU.")
    model.to(device)
    criterion.to(device)

    # Set up TensorBoard writer
    # writer = SummaryWriter()
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run{now}"
    wandb.init(project='catbert-test', name=run_name, dir='/home/jovyan/shared-scratch/jhoon/CATBERT/log')
    wandb.watch(model, log="all")


    # Train the model
    for epoch in range(1, num_epochs + 1):
        loss = train(model, optimizer, train_loader, criterion, device)
        print(f"Epoch {epoch} loss: {loss:.4f}")
        #writer.add_scalar("Train/Loss", loss, epoch)
        if epoch%5 == 0:
            wandb.log({"loss": loss})
        
            
    torch.save(model.state_dict(), f"./checkpoint/pretrain/ckpt_{now}.pt")
    wandb.finish()



"""
Write a training script for training the model on the dataset. You can use the following snippet as a starting point:

"""



import numpy as np
import torch, transformers, os
import torch.nn as nn
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from model.classifiers import BinaryClassifier, MultiLabelClassifier
from model.dataset import PretrainDataset
import wandb
from tqdm import tqdm
import datetime


# def criterion(loss_fn, outputs, pictures, device):
#   losses = 0
#   for i, key in enumerate(outputs):
#     losses += loss_fn(outputs[key], pictures['labels'][f'{key}'].to(device))
#   return losses

def class_acc(n_correct,n_samples,n_class_correct,n_class_samples,class_list):
    for i in range(len(class_list)):
      print("-------------------------------------------------")
      acc = 100.0 * n_correct[i] / n_samples
      print(f'Overall class performance: {round(acc,1)} %')
      for k in range(len(class_list[i])):
          acc = 100.0 * n_class_correct[i][k] / n_class_samples[i][k]
          print(f'Accuracy of {class_list[i][k]}: {round(acc,1)} %')
    print("-------------------------------------------------")


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    lr_list = []
    train_losses = []
    print('training...')
    for batch in tqdm(data_loader):
        ids = batch["ids"].to(device, dtype=torch.long)
        masks = batch["masks"].to(device, dtype=torch.long)
        labels = batch["labels"] #.to(device)

        optimizer.zero_grad()
        outputs = model(ids, masks) # need to set the shape of output
        # compute loss
        loss = 0
        for key in outputs:
           loss += nn.CrossEntropyLoss()(outputs[key], labels[key].to(device))
        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()
        lr_list.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return np.mean(train_losses), np.mean(lr_list)

def validate_fn(data_loader, model, device, n_class_type=3):
    model.eval()
    val_losses = []
    val_overall_acc = []
    val_class_acc = []
    n_correct = torch.zeros(n_class_type) #[]
    n_samples = 0
    print('validating...')
    with torch.no_grad():
        for batch in tqdm(data_loader):
            ids = batch["ids"].to(device, dtype=torch.long)
            masks = batch["masks"].to(device, dtype=torch.long)
            labels = batch["labels"] #.to(device, dtype=torch.float)
            outputs = model(ids, masks)
            loss = 0
            
            for i, key in enumerate(outputs):
                loss += nn.CrossEntropyLoss()(outputs[key], labels[key].to(device))
                _, predicted = torch.max(outputs[key], 1)
                label_logit = labels[key].argmax(dim=1).to(device)
                # import pdb; pdb.set_trace()
                n_correct[i] += (predicted == label_logit).sum().item() 
                
                if i == 0:
                    n_samples += labels[key].size(0)
                  
            
            val_losses.append(loss.item())
        accuracy = n_correct/n_samples
        val_overall_acc = torch.mean(accuracy).item()
        val_class_acc = dict(zip(labels.keys(), accuracy.tolist()))
    
    return np.mean(val_losses), val_overall_acc, val_class_acc


def run_pretraining(df_train, df_val, params, model, tokenizer, device, run_name):
    EPOCHS = params['num_epochs']
    EARLY_STOP_THRESHOLD = params['early_stop_threshold']
    TRAIN_BS = params['batch_size']
    VAL_BS = TRAIN_BS 
    LR = params["lr"] if params.get("lr") else 1e-6 # Learning rate
    WRMUP = params["warmup_steps"] if params.get("warmup_steps") else 0
    SCHD = params["scheduler"] if params.get("scheduler") else "linear" # scheduler type
    #HEAD = params["model_head"] if params.get("model_head") else "multilabel"

    print("=============================================================")
    print(f"{run_name} is launched")
    print("=============================================================")

    wandb.init(project="catbert-pt-class", name=run_name, dir='./log')
               #dir='/home/jovyan/shared-scratch/jhoon/CATBERT/log')

    #=========================================================================
    # Prepare data and model for training
    #=========================================================================   
    # Initialize training dataset
    train_dataset = PretrainDataset(texts = df_train["text"].values,
                                    ads_size = df_train["ads_size"].values,
                                    ads_class = df_train["ads_class"].values,
                                    bulk_class = df_train["bulk_class"].values,
                                    tokenizer = tokenizer,
                                    seq_len= tokenizer.model_max_length)
    # Initialize validation dataset
    val_dataset = PretrainDataset(texts = df_val["text"].values,
                                  ads_size = df_val["ads_size"].values,
                                  ads_class = df_val["ads_class"].values,
                                  bulk_class = df_val["bulk_class"].values,
                                  tokenizer = tokenizer,
                                  seq_len= tokenizer.model_max_length)
    
    # Create training dataloader
    train_data_loader = DataLoader(train_dataset, batch_size = TRAIN_BS,
                                   shuffle = True, num_workers = 2)
    # Create validation dataloader
    val_data_loader = DataLoader(val_dataset, batch_size = VAL_BS,
                                 shuffle = False, num_workers = 2)
    
    # Load model and send it to the device.
    model = model.to(device) 

    wandb.watch(model, log="all")
    # Get the AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    # Calculate the number of training steps (this is used by scheduler).
    # training steps = [number of batches] x [number of epochs].
    train_steps = int(len(df_train) / TRAIN_BS * EPOCHS)    
    # Get the learning rate scheduler    
    scheduler = transformers.get_scheduler(
                    SCHD,    # Create a schedule with a learning rate that decreases linearly 
                                 # from the initial learning rate set in the optimizer to 0.
                    optimizer = optimizer,
                    num_warmup_steps = WRMUP, #50
                    num_training_steps = train_steps)
    best_loss = 999
    early_stopping_counter = 0       
    for epoch in range(1, EPOCHS+1):
        # Call the train function and get the training loss
        train_loss, lr = train_fn(train_data_loader, model, optimizer, device, scheduler)
        # Perform validation and get the validation loss
        val_loss, val_acc, val_class_acc = validate_fn(val_data_loader, model, device)

        loss = val_loss
        report_dict = {'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc, 'lr': lr}
        report_dict.update(val_class_acc)
        wandb.log(report_dict)
        # If there's improvement on the validation loss, save the model checkpoint.
        # Else do early stopping if threshold is reached.
        if loss < best_loss:            
            save_ckpt_path = os.path.join("./checkpoint/pretrain/", run_name)
            if not os.path.exists(save_ckpt_path):
                os.makedirs(save_ckpt_path)
            torch.save(model.state_dict(), os.path.join(save_ckpt_path, 'checkpoint.pt'))
            print(f"Epoch: {epoch}, Train Loss = {round(train_loss,3)}, Val Loss = {round(val_loss,3)}, Val Acc = {round(val_acc,3)}, checkpoint saved.")
            best_loss = loss
            early_stopping_counter = 0
        else:
            print(f"Epoch: {epoch}, Train Loss = {round(train_loss,3)}, Val Loss = {round(val_loss,3)}, Val Acc = {round(val_acc,3)}")
            early_stopping_counter += 1
        if early_stopping_counter > EARLY_STOP_THRESHOLD:
            print(f"Early stopping triggered at epoch {epoch}! Best Loss: {round(best_loss,3)}\n")                
            break

    print(f"===== Training Termination =====")        
    wandb.finish()
import numpy as np
import torch, transformers, os
import torch.nn as nn
from torch.utils.data import DataLoader
from model.dataset import MultimodalDataset
import wandb
from tqdm import tqdm



def mae_loss_fn(predictions, targets):
    return torch.mean(torch.abs(targets - predictions))

def rmse_loss_fn(predictions, targets):       
    return torch.sqrt(nn.MSELoss()(predictions, targets))


def train_fn(data_loader, model, optimizer, device, scheduler, loss_fn='rmse'):
    model.train()                               # Put the model in training mode.                   
    lr_list = []
    train_losses = []
    print('training...')
    
    for batch in tqdm(data_loader):                   # Loop over all batches.
        # convert dictionary input to torch tensors
        sections = batch["ids"].keys()
        ids = [] 
        masks = [] 
        for section in sections:
            ids.append(batch["ids"][section])
            masks.append(batch["masks"][section])
        #breakpoint()
        ids = torch.stack(ids, dim=0).to(device, dtype=torch.long) # [3, batch_size, seq_len]
        masks = torch.stack(masks, dim=0).to(device, dtype=torch.long) # [3, batch_size, seq_len]
        targets = batch["target"].to(device, dtype=torch.float) # [batch_size]

        optimizer.zero_grad()                   # To zero out the gradients.
        outputs = model(ids, masks).squeeze(-1) # Predictions from 1 batch of data.
        
        # Get the training loss.
        if loss_fn == 'mae':
            loss = mae_loss_fn(outputs, targets)
        elif loss_fn =='rmse':
            loss  = rmse_loss_fn(outputs, targets)
        elif loss_fn == 'L2':
            loss = nn.MSELoss()(outputs, targets)    
        train_losses.append(loss.item())
        #loss.backward(retain_graph=True)        
        loss.backward()                         # To backpropagate the error (gradients are computed).
        optimizer.step()                        # To update parameters based on current gradients.
        lr_list.append(optimizer.param_groups[0]["lr"])
        scheduler.step()                        # To update learning rate.    
    return np.mean(train_losses), np.mean(lr_list)

def validate_fn(data_loader, model, device, loss_fn='rmse'):  
    model.eval()                                    # Put model in evaluation mode.
    val_losses = []
    val_maes = []
    print('validating..')
    with torch.no_grad():                           # Disable gradient calculation.
        for batch in tqdm(data_loader):                   # Loop over all batches.   
            
            sections = batch["ids"].keys()
            ids = [] 
            masks = [] 
            for section in sections:
                ids.append(batch["ids"][section])
                masks.append(batch["masks"][section])
                # targets = torch.cat((targets, batch["target"][section]), 0)
            ids = torch.stack(ids, dim=0).to(device, dtype=torch.long)
            masks = torch.stack(masks, dim=0).to(device, dtype=torch.long)
            targets = batch["target"].to(device, dtype=torch.float)

            outputs = model(ids, masks).squeeze(-1) # Predictions from 1 batch of data.
             # Get the validation loss.
            if loss_fn == 'mae':
                loss = mae_loss_fn(outputs, targets)
                mae = loss       
            elif loss_fn =='rmse':
                loss  = rmse_loss_fn(outputs, targets)
                mae = mae_loss_fn(outputs, targets)
            elif loss_fn == 'L2':
                loss = nn.MSELoss()(outputs, targets)
                mae = mae_loss_fn(outputs, targets)                               
            val_losses.append(loss.item())
            val_maes.append(mae.item())           
    return np.mean(val_losses), np.mean(val_maes) 


def run_finetuning(df_train, df_val, params, model, tokenizer, device, run_name):  

    # Hyperparameters and settings   
    EPOCHS = params["num_epochs"]
    EARLY_STOP_THRESHOLD = params["early_stop_threshold"]  # Set the early stopping threshold    
    TRAIN_BS = params["batch_size"]  # Training batch size
    VAL_BS = TRAIN_BS            # Validation batch size
    LR = params["lr"] if params.get("lr") else 1e-6 # Learning rate
    WRMUP = params["warmup_steps"] if params.get("warmup_steps") else 0 # warmup step for scheduler
    SCHD = params["scheduler"] if params.get("scheduler") else "linear" # scheduler type
    #HEAD = params["model_head"] if params.get("model_head") else "pooler" # Attach model head for regression
    LOSSFN = params["loss_fn"] if params.get("loss_fn") else "rmse" # Attach model head for regression
    
    # ========================================================================
    # Prepare logging and saving path
    # ========================================================================

    print("=============================================================")
    print(f"{run_name} is launched")
    print("=============================================================")
    print(f"Model head: {params['model_head']}")
    print(f"Epochs: {EPOCHS}")
    print(f"Early stopping threshold: {EARLY_STOP_THRESHOLD}")
    print(f"Training batch size: {TRAIN_BS}")
    print(f"Validation batch size: {VAL_BS}")
    print(f"Initial learning rate: {LR}")
    print(f"Warmup steps: {WRMUP}")
    print(f"Scheduler: {SCHD}")
    print(f"Loss function: {LOSSFN}")
    print("=============================================================")
    wandb.init(project="catbert-ft-multi", name=run_name, dir='./log')
               #dir='/home/jovyan/shared-scratch/jhoon/CATBERT/log')
    
    #=========================================================================
    # Prepare data and model for training
    #=========================================================================   
    # Initialize training dataset
    train_dataset = MultimodalDataset(texts_sys = df_train['system'].values,
                                   texts_ads = df_train['adsorbate'].values,
                                   texts_bulk = df_train['bulk'].values,
                                   targets = df_train["target"].values,
                                   tokenizer = tokenizer,
                                   seq_len= tokenizer.model_max_length)
    # Initialize validation dataset
    val_dataset = MultimodalDataset(texts_sys = df_val['system'].values,
                                   texts_ads = df_val['adsorbate'].values,
                                   texts_bulk = df_val['bulk'].values,
                                   targets = df_val["target"].values,
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
    if params.get("optimizer") == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01) #originally 1e-6
    # elif params.get("optimizer") == "gLLRD":
    #     optimizer, _ = roberta_base_AdamW_grouped_LLRD(model, LR)
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
    #=========================================================================
    # Training Loop - Start training the epochs
    #=========================================================================   
    best_loss = 999
    early_stopping_counter = 0       
    for epoch in range(1, EPOCHS+1):
        # Call the train function and get the training loss
        train_loss, lr = train_fn(train_data_loader, model, optimizer, device, scheduler, LOSSFN)
        # Perform validation and get the validation loss
        val_loss, val_mae = validate_fn(val_data_loader, model, device, LOSSFN)

        loss = val_loss
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae,'lr': lr})
        # If there's improvement on the validation loss, save the model checkpoint.
        # Else do early stopping if threshold is reached.
        if loss < best_loss:            
            save_ckpt_path = os.path.join("./checkpoint/finetune/", run_name)
            if not os.path.exists(save_ckpt_path):
                os.makedirs(save_ckpt_path)
            torch.save(model.state_dict(), os.path.join(save_ckpt_path, 'checkpoint.pt'))
            print(f"Epoch: {epoch}, Train Loss = {round(train_loss,3)}, Val Loss = {round(val_loss,3)}, Val MAE = {round(val_mae,3)}, checkpoint saved.")
            best_loss = loss
            early_stopping_counter = 0
        else:
            print(f"Epoch: {epoch}, Train Loss = {round(train_loss,3)}, Val Loss = {round(val_loss,3)}, Val MAE = {round(val_mae,3)}")
            early_stopping_counter += 1
        if early_stopping_counter > EARLY_STOP_THRESHOLD:
            print(f"Early stopping triggered at epoch {epoch}! Best Loss: {round(best_loss,3)}\n")                
            break

    print(f"===== Training Termination =====")        
    wandb.finish()



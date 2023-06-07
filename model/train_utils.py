import numpy as np
import torch, transformers, os
import torch.nn as nn
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from model.regressors import MyModel, MyModel2, MyModel_MLP, MyModel_AttnHead, MyModel_ConcatLast4Layers
from model.dataset import MyDataset
import wandb
from tqdm import tqdm
import datetime

def mae_loss_fn(predictions, targets):
    return torch.mean(torch.abs(targets - predictions))

def rmse_loss_fn(predictions, targets):       
    return torch.sqrt(nn.MSELoss()(predictions, targets))

def plot_lr_schedule(learning_rates, save_file_path):
    x = np.arange(len(learning_rates))
    plt.plot(x, learning_rates)
    plt.title(f'Linear schedule')
    plt.ylabel("Learning Rate")
    plt.xlabel("Training Steps")
    plt.savefig(os.path.join(save_file_path, "lr_schedule.png"))

def plot_train_val_losses(train_losses, val_losses, fold, save_file_path):
    x = np.arange(len(train_losses))
    plt.plot(x, train_losses, label="training loss", marker='o')
    plt.plot(x, val_losses, label="validation loss", marker='o')
    plt.legend(loc="best")   # to show the labels.
    plt.title(f'Fold {fold}')    
    plt.ylabel("Loss")
    plt.xlabel(f"Epoch")    
    plt.savefig(os.path.join(save_file_path, f"fold_{fold}_loss.png"))

def roberta_base_AdamW_grouped_LLRD(model, debug=False):
        
    opt_parameters = [] # To be passed to the optimizer (only parameters of the layers you want to update).
    debug_param_groups = []
    named_parameters = list(model.named_parameters()) 
    
    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    set_2 = ["layer.4", "layer.5", "layer.6", "layer.7"]
    set_3 = ["layer.8", "layer.9", "layer.10", "layer.11"]
    init_lr = 1e-6
    
    for i, (name, params) in enumerate(named_parameters):  
        
        weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01
 
        if name.startswith("roberta_model.embeddings") or name.startswith("roberta_model.encoder"):            
            # For first set, set lr to 1e-6 (i.e. 0.000001)
            lr = init_lr       
            
            # For set_2, increase lr to 0.00000175
            lr = init_lr * 1.75 if any(p in name for p in set_2) else lr
            
            # For set_3, increase lr to 0.0000035 
            lr = init_lr * 3.5 if any(p in name for p in set_3) else lr
            
            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})  
            
        # For regressor and pooler, set lr to 0.0000036 (slightly higher than the top layer).                
        if name.startswith("regressor") or name.startswith("roberta_model.pooler"):               
            lr = init_lr * 3.6 
            
            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})    
            
        debug_param_groups.append(f"{i} {name}")

    if debug: 
        for g in range(len(debug_param_groups)): print(debug_param_groups[g]) 

    return torch.optim.AdamW(opt_parameters, lr=init_lr), debug_param_groups

def train_fn(data_loader, model, optimizer, device, scheduler, loss_fn='rmse'):
    model.train()                               # Put the model in training mode.                   
    lr_list = []
    train_losses = []
    print('training')
    for batch in tqdm(data_loader):                   # Loop over all batches.

        ids = batch["ids"].to(device, dtype=torch.long)
        masks = batch["masks"].to(device, dtype=torch.long)
        targets = batch["target"].to(device, dtype=torch.float) 
        optimizer.zero_grad()                   # To zero out the gradients.
        outputs = model(ids, masks).squeeze(-1) # Predictions from 1 batch of data.
        # Get the training loss.
        if loss_fn == 'mae':
            loss = mae_loss_fn(outputs, targets)
        elif loss_fn =='rmse':
            loss  = rmse_loss_fn(outputs, targets)    
        train_losses.append(loss.item())
        loss.backward(retain_graph=True)        
        #loss.backward()                         # To backpropagate the error (gradients are computed).
        optimizer.step()                        # To update parameters based on current gradients.
        lr_list.append(optimizer.param_groups[0]["lr"])
        scheduler.step()                        # To update learning rate.    
    return train_losses, lr_list

def validate_fn(data_loader, model, device, loss_fn='rmse'):  
    model.eval()                                    # Put model in evaluation mode.
    val_losses = []
    val_maes = []
    print('validation')
    with torch.no_grad():                           # Disable gradient calculation.
        for batch in tqdm(data_loader):                   # Loop over all batches.   
            
            ids = batch["ids"].to(device, dtype=torch.long)
            masks = batch["masks"].to(device, dtype=torch.long)
            targets = batch["target"].to(device, dtype=torch.float)
            outputs = model(ids, masks).squeeze(-1) # Predictions from 1 batch of data.
             # Get the validation loss.
            if loss_fn == 'mae':
                mae = mae_loss_fn(outputs, targets)
                loss = mae        
            elif loss_fn =='rmse':
                mae = mae_loss_fn(outputs, targets)
                loss  = rmse_loss_fn(outputs, targets)            
            val_losses.append(loss.item())
            val_maes.append(mae.item())           
    return val_losses, val_maes 


def run_training(df_train, df_val, params, model, tokenizer, device, run_name):  
    
    """
    model_head: Accepted option is "pooler", "attnhead", or "concatlayer"
    """    
    EPOCHS = params["num_epochs"]
    EARLY_STOP_THRESHOLD = params["early_stop_threshold"]  # Set the early stopping threshold    
    TRAIN_BS = params["batch_size"]  # Training batch size
    VAL_BS = TRAIN_BS            # Validation batch size
    warmup_steps = params["warmup_steps"] if params.get("warmup_steps") else 0 # warmup step for scheduler
    model_head = params["model_head"] if params.get("model_head") else "pooler" # Attach model head for regression
    loss_fn = params["loss_fn"] if params.get("loss_fn") else "rmse" # Attach model head for regression
    # ========================================================================
    # Prepare logging and saving path
    # ========================================================================
    #now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print("=========================================")
    print(f"{run_name} is launched")
    print("=========================================")
    # run_name = f"{model_head}_{loss_fn}_{np.round(len(df_train)/1000)}k"
    # if 'args' in kwargs.keys():
    #     run_name += '_'+kwargs['args']
    
    wandb.init(project="catbert-ft", name=run_name, dir='/home/jovyan/shared-scratch/jhoon/CATBERT/log')
    
    #=========================================================================
    # Prepare data and model for training
    #=========================================================================   
    # Initialize training dataset
    train_dataset = MyDataset(texts = df_train["text"].values,
                              targets = df_train["target"].values,
                              tokenizer = tokenizer,
                              seq_len= tokenizer.model_max_length-2)
    # Initialize validation dataset
    val_dataset = MyDataset(texts = df_val["text"].values,
                            targets = df_val["target"].values,
                            tokenizer = tokenizer,
                            seq_len= tokenizer.model_max_length-2)
    # Create training dataloader
    train_data_loader = DataLoader(train_dataset, batch_size = TRAIN_BS,
                                   shuffle = True, num_workers = 2)
    # Create validation dataloader
    val_data_loader = DataLoader(val_dataset, batch_size = VAL_BS,
                                 shuffle = False, num_workers = 2)

    # Load model and send it to the device.        
    if model_head == "pooler":
        model = MyModel(model).to(device) 
    elif model_head == "mlp":
        model = MyModel_MLP(model).to(device)
    elif model_head == "attnhead":
        model = MyModel_AttnHead(model).to(device)
    elif model_head == "concatlayer":
        model = MyModel_ConcatLast4Layers(model).to(device)
    else:
        raise ValueError(f"Unknown model_head: {model_head}") 
    wandb.watch(model, log="all")
    # Get the AdamW optimizer
    if params.get("optimizer") == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7) #originally 1e-6
    elif params.get("optimizer") == "gLLRD":
        optimizer, _ = roberta_base_AdamW_grouped_LLRD(model)
    # Calculate the number of training steps (this is used by scheduler).
    # training steps = [number of batches] x [number of epochs].
    train_steps = int(len(df_train) / TRAIN_BS * EPOCHS)    
    # Get the learning rate scheduler    
    scheduler = transformers.get_scheduler(
                    "linear",    # Create a schedule with a learning rate that decreases linearly 
                                 # from the initial learning rate set in the optimizer to 0.
                    optimizer = optimizer,
                    num_warmup_steps = warmup_steps, #50
                    num_training_steps = train_steps)
    #=========================================================================
    # Training Loop - Start training the epochs
    #=========================================================================   
    best_loss = 999
    early_stopping_counter = 0       
    for epoch in range(1, EPOCHS+1):
        # Call the train function and get the training loss
        train_losses, lr_list = train_fn(train_data_loader, model, optimizer, device, scheduler, loss_fn)
        train_loss = np.mean(train_losses)   
        # Perform validation and get the validation loss
        val_losses, val_maes = validate_fn(val_data_loader, model, device, loss_fn)
        val_loss = np.mean(val_losses)
        val_mae = np.mean(val_maes) 
        loss = val_loss
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae,'lr': lr_list[-1]})
        # If there's improvement on the validation loss, save the model checkpoint.
        # Else do early stopping if threshold is reached.
        if loss < best_loss:            
            save_ckpt_path = os.path.join("./checkpoint/finetune/", run_name+".pt")
            print(save_ckpt_path)
            # if not os.path.exists(save_ckpt_path):
            #     os.makedirs(save_ckpt_path)
            torch.save(model.state_dict(), save_ckpt_path)
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


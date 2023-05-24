import numpy as np
import torch, transformers, os
import torch.nn as nn
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from model.regressors import MyModel, MyModel_AttnHead, MyModel_ConcatLast4Layers
from model.dataset import MyDataset

def mae_loss_fn(prediction, target):
    return torch.mean(torch.abs(target - prediction))

def rmse_loss_fn(predictions, targets):       
    return torch.sqrt(nn.MSELoss()(predictions, targets))

def plot_lr_schedule(learning_rates, save_file_path):
    x = np.arange(len(learning_rates))
    plt.plot(x, learning_rates)
    plt.title(f'Linear schedule')
    plt.ylabel("Learning Rate")
    plt.xlabel("Training Steps")
    #plt.show()
    plt.savefig(os.path.join(save_file_path, "lr_schedule.png"))

def plot_train_val_losses(train_losses, val_losses, fold, save_file_path):
    x = np.arange(len(train_losses))
    plt.plot(x, train_losses, label="training loss", marker='o')
    plt.plot(x, val_losses, label="validation loss", marker='o')
    plt.legend(loc="best")   # to show the labels.
    plt.title(f'Fold {fold}')    
    plt.ylabel("Loss")
    plt.xlabel(f"Epoch")    
    #plt.show()
    plt.savefig(os.path.join(save_file_path, f"fold_{fold}_loss.png"))

def train_fn(data_loader, model, optimizer, device, scheduler, loss='mae'):
    model.train()                               # Put the model in training mode.                   
    lr_list = []
    train_losses = []

    for batch in data_loader:                   # Loop over all batches.

        ids = batch["ids"].to(device, dtype=torch.long)
        masks = batch["masks"].to(device, dtype=torch.long)
        targets = batch["target"].to(device, dtype=torch.float) 
        optimizer.zero_grad()                   # To zero out the gradients.
        outputs = model(ids, masks).squeeze(-1) # Predictions from 1 batch of data.
        # Get the training loss.
        if loss == 'mae':
            loss = mae_loss_fn(outputs, targets)        
        elif loss =='rmse':
            loss  = rmse_loss_fn(outputs, targets)    
        train_losses.append(loss.item())
        loss.backward(retain_graph=True)        
        #loss.backward()                         # To backpropagate the error (gradients are computed).
        optimizer.step()                        # To update parameters based on current gradients.
        lr_list.append(optimizer.param_groups[0]["lr"])
        scheduler.step()                        # To update learning rate.    
    return train_losses, lr_list

def validate_fn(data_loader, model, device, loss='mae'):  
    model.eval()                                    # Put model in evaluation mode.
    val_losses = []
   
    with torch.no_grad():                           # Disable gradient calculation.
        for batch in data_loader:                   # Loop over all batches.   
            
            ids = batch["ids"].to(device, dtype=torch.long)
            masks = batch["masks"].to(device, dtype=torch.long)
            targets = batch["target"].to(device, dtype=torch.float)
            outputs = model(ids, masks).squeeze(-1) # Predictions from 1 batch of data.
             # Get the validation loss.
            if loss == 'mae':
                loss = mae_loss_fn(outputs, targets)        
            elif loss =='rmse':
                loss  = rmse_loss_fn(outputs, targets)            
            val_losses.append(loss.item())           
    return val_losses 


def run_training(df, tokenizer, model_head="pooler"):  
    
    """
    model_head: Accepted option is "pooler", "attnhead", or "concatlayer"
    """    
    EPOCHS = 5
    FOLDS = [0, 1, 2, 3, 4]   # Set the list of folds you want to train
    EARLY_STOP_THRESHOLD = 3    
    TRAIN_BS = 16             # Training batch size     
    VAL_BS = 64               # Validation batch size  
    cv = []                   # A list to hold the cross validation scores
    save_file_path = os.path.join("./results/", f"roberta_base_{model_head}")
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    #=========================================================================
    # Prepare data and model for training
    #=========================================================================
    
    for fold in FOLDS:
        #set_random_seed(3377)
        # Initialize the tokenizer
        #tokenizer =  RobertaTokenizer.from_pretrained("roberta-base")
        # Fetch training data
        df_train = df[df["skfold"] != fold].reset_index(drop=True)

        # Fetch validation data
        df_val = df[df["skfold"] == fold].reset_index(drop=True)

        # Initialize training dataset
        train_dataset = MyDataset(texts = df_train["text"].values,
                                  targets = df_train["target"].values,
                                  tokenizer = tokenizer)

        # Initialize validation dataset
        val_dataset = MyDataset(texts = df_val["text"].values,
                                targets = df_val["target"].values,
                                tokenizer = tokenizer)

        # Create training dataloader
        train_data_loader = DataLoader(train_dataset, batch_size = TRAIN_BS,
                                       shuffle = True, num_workers = 2)

        # Create validation dataloader
        val_data_loader = DataLoader(val_dataset, batch_size = VAL_BS,
                                     shuffle = False, num_workers = 2)

        # Initialize the cuda device (or use CPU if you don't have GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        # Load model and send it to the device.        
        if model_head == "pooler":
            model = MyModel().to(device)
        elif model_head == "attnhead":
            model = MyModel_AttnHead().to(device)
        elif model_head == "concatlayer":
            model = MyModel_ConcatLast4Layers().to(device)
        else:
            raise ValueError(f"Unknown model_head: {model_head}") 

        # Get the AdamW optimizer
        optimizer = transformers.AdamW(model.parameters(), lr=1e-6)

        # Calculate the number of training steps (this is used by scheduler).
        # training steps = [number of batches] x [number of epochs].
        train_steps = int(len(df_train) / TRAIN_BS * EPOCHS)    

        # Get the learning rate scheduler    
        scheduler = transformers.get_scheduler(
                        "linear",    # Create a schedule with a learning rate that decreases linearly 
                                     # from the initial learning rate set in the optimizer to 0.
                        optimizer = optimizer,
                        num_warmup_steps = 0,
                        num_training_steps = train_steps)

        #=========================================================================
        # Training Loop - Start training the epochs
        #=========================================================================

        print(f"===== FOLD: {fold} =====")    
        best_loss = 999
        early_stopping_counter = 0       
        all_train_losses = []
        all_val_losses = []
        all_lr = []

        for epoch in range(EPOCHS):

            # Call the train function and get the training loss
            train_losses, lr_list = train_fn(train_data_loader, model, optimizer, device, scheduler)
            train_loss = np.mean(train_losses)   
            all_train_losses.append(train_loss)
            all_lr.extend(lr_list)

            # Perform validation and get the validation loss
            val_losses = validate_fn(val_data_loader, model, device)
            val_loss = np.mean(val_losses)
            all_val_losses.append(val_loss)    

            loss = val_loss

            # If there's improvement on the validation loss, save the model checkpoint.
            # Else do early stopping if threshold is reached.
            if loss < best_loss:            
                torch.save(model.state_dict(), os.path.join(save_file_path, f"fold_{fold}_ckpt.pt"))
                print(f"FOLD: {fold}, Epoch: {epoch}, Loss = {round(loss,4)}, checkpoint saved.")
                best_loss = loss
                early_stopping_counter = 0
            else:
                print(f"FOLD: {fold}, Epoch: {epoch}, Loss = {round(loss,4)}")
                early_stopping_counter += 1
            if early_stopping_counter > EARLY_STOP_THRESHOLD:
                print(f"FOLD: {fold}, Epoch: {epoch}, Loss = {round(loss,4)}")
                print(f"Early stopping triggered! Best Loss: {round(best_loss,4)}\n")                
                break

        # Plot the losses and learning rate schedule.
        plt.clf()
        plot_train_val_losses(all_train_losses, all_val_losses, fold, save_file_path)
        plt.clf()
        plot_lr_schedule(all_lr, save_file_path)   
        
        # Keep the best_loss as cross validation score for the fold.
        cv.append(best_loss)
        
    # Print the cross validation scores and their average.
    cv_rounded = [ round(elem, 4) for elem in cv ] 
    print(f"CV: {cv_rounded}") 
    print(f"Average CV: {round(np.mean(cv), 4)}\n")
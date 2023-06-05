from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM
from pathlib import Path
from model.dataset import stratified_kfold 
import torch
from torch.utils.data import Dataset, DataLoader
# from transformers import AdamW
from tqdm import tqdm 
import wandb
import datetime
import pandas as pd
import os
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# mask language modeling
def mlm(tensor, mask_token_id=4):
    rand = torch.rand(tensor.shape) #[0,1]
    mask_arr = (rand < 0.15) * (tensor > 2)
    for i in range(tensor.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero())
        tensor[i, selection] = mask_token_id #tokenizer.mask_token_id
    return tensor 

# convert files to input tensors
# def get_encodings(paths):
#     input_ids = []
#     mask = []
#     labels = []
#     for path in tqdm(paths):
#         with open(path, 'r', encoding='utf-8') as f:
#             text = f.read()
#         sample = tokenizer(text, max_length=512, padding='max_length', 
#                            truncation=True, return_tensors='pt')
#         labels.append(sample['input_ids'])
#         mask.append(sample['attention_mask'])
#         input_ids.append(mlm(sample['input_ids'].clone()))
#     input_ids = torch.cat(input_ids)
#     mask = torch.cat(mask)
#     labels = torch.cat(labels)
#     return {'input_ids': input_ids, 
#             'attention_mask': mask, 
#             'labels': labels}

# convert text to input tensors
def get_encodings_from_texts(texts, tokenizer):
    input_ids = []
    mask = []
    labels = []
    max_len = tokenizer.model_max_length
    for text in tqdm(texts):
        # substract 2 for <s> and </s> from max_lens
        sample = tokenizer(text, max_length= max_len-2, padding='max_length', 
                           truncation=True, return_tensors='pt')
        labels.append(sample['input_ids'])
        mask.append(sample['attention_mask'])
        input_ids.append(mlm(sample['input_ids'].clone()))
    input_ids = torch.cat(input_ids)
    mask = torch.cat(mask)
    labels = torch.cat(labels)
    return {'input_ids': input_ids, 
            'attention_mask': mask, 
            'labels': labels}

# train function
def train(model, dataloader, optim, device, mode):
    if mode == 'train':    
        model.train()
        print('training...')
    elif mode == 'val':
        model.eval()
        print('validating...')
    loop = tqdm(dataloader, leave=True)
    total_loss = 0
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# roberta dataset class
class RobertaDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings['input_ids'])


def run_pretraining(df_train, df_val, tokenizer, model, device, params, save_path):
    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"]
    lr = params["lr"]
    early_stop_threshold = params["early_stop_threshold"]
    # ===============================================
    # Set run name and initialize wandb
    # ===============================================
    now = datetime.datetime.now().strftime('%m%d_%H%M')
    run_name = f'len{tokenizer.model_max_length}_ep{num_epochs}_bs{batch_size}_{now}'
    wandb.init(project="catbert-pt", name=run_name, dir='/home/jovyan/shared-scratch/jhoon/CATBERT/log')
    

    # ===============================================
    # Prepare data and model for training
    # ===============================================
    train_texts = df_train['text'].values.tolist()
    val_texts = df_val['text'].values.tolist()

    train_encodings = get_encodings_from_texts(train_texts, tokenizer)
    val_encodings = get_encodings_from_texts(val_texts, tokenizer)
    
    train_dataset = RobertaDataset(train_encodings)
    val_dataset = RobertaDataset(val_encodings)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  
    # ===============================================
    # Training loop
    # ===============================================
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    best_loss = 999
    early_stopping_counter = 0
    for epoch in range(1, num_epochs+1):
        train_loss = train(model, train_dataloader, optim, device, mode='train')
        val_loss = train(model, val_dataloader, optim, device, mode='val')
        loss = val_loss
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, 'lr': lr})
        if loss < best_loss:
            best_loss = loss
            early_stopping_counter = 0
            #torch.save(model.state_dict(), f'./checkpoint/pretrain/{run_name}.pt')
            full_save_path = os.path.join(save_path, run_name)
            if not os.path.exists(full_save_path):
                os.makedirs(full_save_path)
            model.save_pretrained(full_save_path)
            print(f"Epoch: {epoch}, Train Loss = {round(train_loss,3)}, Val Loss = {round(val_loss,3)}, checkpoint saved.")
        else:
            early_stopping_counter += 1
            print(f"Epoch: {epoch}, Train Loss = {round(train_loss,3)}, Val Loss = {round(val_loss,3)}, no improvement.")

        if early_stopping_counter >= early_stop_threshold:
            print(f"Early stopping at epoch {epoch}! Best loss: {round(best_loss,3)}\n")
            break

    print(f"===== Training Termination =====") 
    wandb.finish()



if __name__ == '__main__':
    # ==========================================================
    # Set up paths for data, model, tokenizer, and results
    # ==========================================================
    train_data_path = "./data/df_is2re_100k.pkl"
    val_data_path = "./data/df_is2re_val_25k.pkl"
    config_path = "./config/roberta_config.json"
    tknz_path = "./tokenizer"
    # ==========================================================
    # Load config and hyperparameters
    with open(config_path, "r") as f:
        loaded_dict = json.load(f)
    
    
    # Hyperparameters for training
    params = loaded_dict['pretrain_params']
   
    # Load training/validation data w/o stratified kfold
    df_train = pd.read_pickle(train_data_path)
    df_val = pd.read_pickle(train_data_path)
    # Load training data w/ stratified kfold
    # df = pd.read_pickle(train_data_path)
    # df = df.iloc[:10000] # for code testing
    # df = stratified_kfold(df, n_splits=5)
    # df_train = df[df['skfold']!=0]
    # df_val = df[df['skfold']==0]

    # Load model
    config = RobertaConfig.from_dict(loaded_dict["roberta_config"])
    model = RobertaForMaskedLM(config)
        
    # Load pre-trained tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(tknz_path, 
                                                     max_len=config.max_position_embeddings) #orginally set as 514

    
    # Set device    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')    
    
    # run pretraining
    run_pretraining(df_train, df_val, 
                    tokenizer, model, 
                    device, params,
                    save_path="./checkpoint/pretrain/")


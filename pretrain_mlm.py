from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM
from pathlib import Path
from model.dataset import stratified_kfold 
import torch
from torch.utils.data import Dataset, DataLoader
# from transformers import AdamW
from tqdm import tqdm 
import wandb
from datetime import datetime
import pandas as pd
import os, yaml, shutil
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
def train(model, dataloader, optim, device):    
    model.train()
    print('training...')
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


def validate(model, dataloader, device):
    model.eval()
    print('validating...')
    loop = tqdm(dataloader, leave=True)
    total_loss = 0
    with torch.no_grad():
        for batch in loop:
        
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
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


def run_pretraining(df_train, df_val, params, model, tokenizer, device, run_name):
    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"]
    lr = params["lr"]
    early_stop_threshold = params["early_stop_threshold"]
    # ===============================================
    # Set run name and initialize wandb
    # ===============================================
    wandb.init(project="catbert-pt", name=run_name, dir='./log')
    #'/home/jovyan/shared-scratch/jhoon/CATBERT/log')
    print("===============================================")
    print(f"Run name: {run_name}")
    print("===============================================")
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
        train_loss = train(model, train_dataloader, optim, device)
        val_loss = validate(model, val_dataloader, device)
        loss = val_loss
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, 'lr': lr})
        if loss < best_loss:
            best_loss = loss
            early_stopping_counter = 0
    

            save_ckpt_path = os.path.join("./checkpoint/pretrain/", run_name)
            if not os.path.exists(save_ckpt_path):
                os.makedirs(save_ckpt_path)
            # model.save_pretrained(full_save_path)
            torch.save(model.state_dict(), os.path.join(save_ckpt_path, 'checkpoint.pt'))
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
    import argparse 
    parser = argparse.ArgumentParser(description='Set the running mode')
    parser.add_argument('--debug', action='store_true', help='Enable debugging mode')
    parser.add_argument('--base', action='store_true', help='Pretrain with base model') 
    args = parser.parse_args()
    if args.debug:
        print('Debugging mode enabled!')
    if args.base:
        print('Pretraining with base model!')

    # ============== 0. Read pretrain config file ======================
    pt_config_path = "./config/pt_config.yaml"
    paths = yaml.load(open(pt_config_path, "r"), Loader=yaml.FullLoader)['paths']
    train_data_path = paths["train_data"]
    val_data_path = paths["val_data"]
    rbt_config_path = paths["roberta_config"]
    tknz_path = paths["tknz"]
    model_config = yaml.load(open(rbt_config_path, 'r'), Loader=yaml.FullLoader)   
    params = yaml.load(open(pt_config_path, "r"), Loader=yaml.FullLoader)['params']
   
    # ================= 1. Load data ======================
    df_train = pd.read_pickle(train_data_path) #.sample(1, random_state=42)
    df_val = pd.read_pickle(val_data_path) #.sample(1, random_state=42)
    if args.debug:
        df_train = df_train.sample(1, random_state=42)
        df_val = df_val.sample(2, random_state=42)
    print("Training data size: " + str(df_train.shape[0]))
    print("Validation data size : " + str(df_val.shape[0]))

    # ================= 2. Load model ======================
    if args.base:
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        #max_len = model.roberta.embeddings.position_embeddings.num_embeddings
    else:
        roberta_config = RobertaConfig.from_dict(model_config)
        model = RobertaForMaskedLM(roberta_config)
        
    
    # ================= 3. Load tokenizer ======================
    max_len = model.roberta.embeddings.position_embeddings.num_embeddings
    tokenizer = RobertaTokenizerFast.from_pretrained(tknz_path, max_len=max_len)
    
    # ================= 4. Set device ======================   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    if args.debug:
        device = torch.device('cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')    
    
    # ================= 5. Run pretraining ======================
    run_name = 'mlm-pt'+datetime.now().strftime("_%m%d_%H%M")
    if args.debug:
        run_name = 'debug'+datetime.now().strftime("_%m%d_%H%M")
    if args.base:
        run_name = 'base-'+run_name
    print("Run name: ", run_name)

    save_dir = os.path.join(f"./checkpoint/pretrain/{run_name}")
    if os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save config files for reference
    shutil.copy(pt_config_path, os.path.join(save_dir, "pt_config.yaml"))
    if not args.base:
        shutil.copy(rbt_config_path, os.path.join(save_dir, "roberta_config.yaml"))

    # run pretraining
    run_pretraining(df_train, df_val, params, model, tokenizer, device, run_name)
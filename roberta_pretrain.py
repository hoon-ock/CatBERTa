from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM
from pathlib import Path
from model.dataset import stratified_kfold 
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm 
import wandb
import datetime
import pandas as pd
import os
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
        # loop.set_description('Epoch: {}'.format(epoch + 1))
        # loop.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)

# roberta dataset class
class RobertaDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings['input_ids'])


def run_pretraining(df_train, df_val, tokenizer, model, device, lr, batch_size=32, num_epochs=5):
    early_stop_threshold = 3
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
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
  
    # ===============================================
    # Training loop
    # ===============================================
    model.to(device)
    optim = AdamW(model.parameters(), lr=lr)
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
            torch.save(model.state_dict(), f'./checkpoint/pretrain/{run_name}.pt')
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
    
    # hyperparameters for training
    num_epochs = 5
    lr = 1e-4
    batch_size = 16
    max_seq_len = 768

    # roberta config
    config = RobertaConfig(
        vocab_size=30_522, #tokenizer.vocab_size,
        max_position_embeddings=max_seq_len, #originally set as 514
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )


    # load pre-trained tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('./tokenizer', max_len=max_seq_len) #orginally set as 514

    # load training files <-- this can be obtained with dataframe
    # paths = [str(x) for x in Path('./data/pre-train').glob('random*.txt')]
    # encodings = get_encodings(paths[:100])
    df = pd.read_pickle('./data/df_train.pkl')
    #df = df.iloc[:10000] # for code testing
    df = stratified_kfold(df, n_splits=5)
    df_train = df[df['skfold']!=0]
    df_val = df[df['skfold']==0]

    # texts = df['text'].values.tolist()
    # encodings = get_encodings_from_texts(texts, tokenizer)
    # dataset = RobertaDataset(encodings)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # load model
    model = RobertaForMaskedLM(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')    
    # model.to(device)
    # optim = AdamW(model.parameters(), lr=lr)
    
    # set up wandb
    # now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # run_name = f'epoch{num_epochs}_bs{batch_size}_{now}' #f"run{now}"
    # wandb.init(project='catbert-pt', name=run_name, dir='/home/jovyan/shared-scratch/jhoon/CATBERT/log')
    # wandb.watch(model, log="all")


    # training loop
    # for epoch in range(num_epochs):
    #     loss = train(model, dataloader, optim, device)
    #     print(f"Epoch: {epoch+1}, Loss: {loss:.4f}")
    #     #model.save_pretrained('model'
    #     if epoch%2 == 0:
    #         wandb.log({"loss": loss})
    
    # torch.save(model.state_dict(), f'./checkpoint/pretrain/{run_name}.pt')
    # wandb.finish()
    run_pretraining(df_train, df_val, 
                    tokenizer, model, 
                    device, lr, 
                    batch_size, num_epochs)


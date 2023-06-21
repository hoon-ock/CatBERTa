import torch
import numpy as np
import pandas as pd
from transformers import RobertaModel, RobertaTokenizerFast, RobertaConfig
from tqdm import tqdm
from model.dataset import FinetuneDataset, PretrainDataset
from model.common import backbone_wrapper
from torch.utils.data import DataLoader
import os, yaml, pickle


def ft_predict_fn(df, model, tokenizer, device):  
    '''
    df: input dataframe with columns "text" and "target" (and "class lables")
    model: model to predict
    tokenizer: tokenizer to tokenize the text
    device: device to run the model
    '''
    dataset = FinetuneDataset(texts = df["text"].values,
                              targets = df["target"].values,
                              tokenizer = tokenizer,
                              seq_len= tokenizer.model_max_length-2)
    
    data_loader = DataLoader(dataset, batch_size=16,
                            shuffle=False, num_workers=2)
    
    
    model.eval()                                    # Put model in evaluation mode.
    print('predicting...')
    predictions = []
    with torch.no_grad():                           # Disable gradient calculation.
        for batch in tqdm(data_loader):                   # Loop over all batches.   
            
            ids = batch["ids"].to(device, dtype=torch.long)
            masks = batch["masks"].to(device, dtype=torch.long)
            targets = batch["target"].to(device, dtype=torch.float)
            outputs = model(ids, masks).squeeze(-1) # Predictions from 1 batch of data.
            #import pdb; pdb.set_trace()
            outputs = outputs.detach().cpu().numpy()
            predictions.extend(outputs.tolist())

    return predictions

# def get_energy_from_text(text, model, tokenizer, device):
#     target = 0 
#     text = np.array([text]).astype(object)
#     data = FinetuneDataset(texts=text, targets=np.array([target]), 
#                            tokenizer=tokenizer, 
#                            seq_len= tokenizer.model_max_length-2)

#     id = data[0]['ids'].unsqueeze(0)
#     mask = data[0]['masks'].unsqueeze(0)
    
#     model.to(device)
#     id = id.to(device)
#     mask = mask.to(device)
#     #import pdb; pdb.set_trace()
#     model.eval()
#     output = model(id, mask).squeeze(-1)
#     return output.item()

if __name__ == '__main__':
    # ================== 0. Load Checkpoint and Configs ==================
    ckpt_dir = "./checkpoint/finetune/ft_0619_1527"
    # Load Training Config
    ft_config = os.path.join(ckpt_dir, "ft_config.yaml")
    paths = yaml.load(open(ft_config, "r"), Loader=yaml.FullLoader)['paths']
    val_data_path = paths["val_data"]
    tknz_path = paths["tknz"]
    head = yaml.load(open(ft_config, "r"), Loader=yaml.FullLoader)['params']['model_head']
    # Load Model Config
    model_config = yaml.load(open(os.path.join(ckpt_dir, "roberta_config.yaml"), "r"), Loader=yaml.FullLoader)
    roberta_config = RobertaConfig.from_dict(model_config)
    # Load Checkpoint
    checkpoint = os.path.join(ckpt_dir, "checkpoint.pt")
    # Set Device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") 

    # ================== 1. Load Model and Tokenizer ==================
    backbone = RobertaModel.from_pretrained('roberta-base', config=roberta_config, ignore_mismatched_sizes=True)
    model = backbone_wrapper(backbone, head)
    model.load_state_dict(torch.load(checkpoint))
    
    max_len = model.roberta_model.embeddings.position_embeddings.num_embeddings
    tokenizer = RobertaTokenizerFast.from_pretrained(tknz_path, max_len=max_len)                                

    # ================== 2. Load Data ==================                               
    df_val = pd.read_pickle(val_data_path)
    # df_val = df_val.sample(5, random_state=42) # for debugging
    # df_val.set_index('id', inplace=True)  
    # ================== 3. Obtain Predictions ==================
    predictions = ft_predict_fn(df_val, model, tokenizer, device)
    
    # save predictions as dictionary with id as key
    save_path = os.path.join(ckpt_dir, "predictions.pkl")
    results = {}
    for i in range(len(df_val)):
        results[df_val.iloc[i]['id']] = predictions[i]
    with open(save_path, "wb") as f:
        pickle.dump(results, f)


    #import pdb; pdb.set_trace()
    # obtain catbert predictions
    # results={}
    # for i in tqdm(df_val.index):
    #     results[i] = get_energy_from_text(df_val.loc[i, 'text'], model, tokenizer, device)
    # save results
    
    import pdb; pdb.set_trace()
    # with open(save_path, "wb") as f:
    #     pickle.dump(results, f)
    

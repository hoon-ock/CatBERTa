import torch
import numpy as np
import pandas as pd
from transformers import RobertaModel, RobertaTokenizerFast, RobertaConfig
from tqdm import tqdm
from model.dataset import FinetuneDataset, PretrainDataset
from model.common import backbone_wrapper, checkpoint_loader
from torch.utils.data import DataLoader
import os, yaml, pickle


def ft_predict_fn(df, model, tokenizer, device, emb_mode=False):  
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
            if not emb_mode:
                # energy prediction
                outputs = model(ids, masks).squeeze(-1) 
            else:
                # embedding generation
                outputs = model(ids, masks)['pooler_output']
            #import pdb; pdb.set_trace()
            outputs = outputs.detach().cpu().numpy()
            predictions.extend(outputs.tolist())

    return predictions


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser(description='Set the running mode')
    parser.add_argument('--emb', action='store_true', help='Embedding generation mode')
    args = parser.parse_args()
    pred = 'energy'
    if args.emb:
        # embedding generation mode
        pred = 'embed'
        print('--------------- Embedding generation mode! ---------------')
    else:
        print('--------------- Energy prediction mode! ---------------')
    
    # ================== 0. Load Checkpoint and Configs ==================
    ckpt_dir = "/home/jovyan/CATBERT/checkpoint/pretrain/pt_0621_0442"
    # Load Training Config
    if 'pretrain' in ckpt_dir:
        train_config = os.path.join(ckpt_dir, "pt_config.yaml")
        pred = 'pt_' + pred
    elif 'finetune' in ckpt_dir:
        train_config = os.path.join(ckpt_dir, "ft_config.yaml")
        pred = 'ft_' + pred
    paths = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)['paths']
    val_data_path = paths["val_data"]
    tknz_path = paths["tknz"]
    head = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)['params']['model_head']
    # Load Model Config
    model_config = yaml.load(open(os.path.join(ckpt_dir, "roberta_config.yaml"), "r"), Loader=yaml.FullLoader)
    roberta_config = RobertaConfig.from_dict(model_config)
    # Load Checkpoint
    checkpoint = os.path.join(ckpt_dir, "checkpoint.pt")
    # Set Device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") # for debugging
    # ================== 1. Load Model and Tokenizer ==================
    backbone = RobertaModel.from_pretrained('roberta-base', config=roberta_config, ignore_mismatched_sizes=True)
    model = backbone_wrapper(backbone, head)
    model.load_state_dict(torch.load(checkpoint))
    max_len = model.roberta_model.embeddings.position_embeddings.num_embeddings

    if args.emb:
        # To generate embeddings, we need to load the model without the head.
        # So, we need to load the model from the checkpoint only on the backbone.
        model = checkpoint_loader(backbone, checkpoint, load_on_roberta=True)
        max_len = model.embeddings.position_embeddings.num_embeddings

    tokenizer = RobertaTokenizerFast.from_pretrained(tknz_path, max_len=max_len)                                
    # ================== 2. Load Data ==================                               
    df_val = pd.read_pickle(val_data_path)
    df_val = df_val.sample(500, random_state=42) # for debugging
    # ================== 3. Obtain Predictions ==================
    predictions = ft_predict_fn(df_val, model, tokenizer, device, emb_mode=args.emb)
    
    # save predictions as dictionary with id as key
    results = {}
    for i in range(len(df_val)):
        results[df_val.iloc[i]['id']] = predictions[i]
    
    save_dir = "/home/jovyan/CATBERT/results/catbert"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"catbert_{pred}_{ckpt_dir.split('/')[-1]}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)   
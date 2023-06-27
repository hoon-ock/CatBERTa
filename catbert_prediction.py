import torch
import numpy as np
import pandas as pd
from transformers import RobertaModel, RobertaTokenizerFast, RobertaConfig
from tqdm import tqdm
from model.dataset import FinetuneDataset, PretrainDataset
from model.common import backbone_wrapper, checkpoint_loader
from torch.utils.data import DataLoader
import os, yaml, pickle


def predict_fn(df, model, tokenizer, device, mode='energy'):  
    '''
    df: input dataframe with columns "text" and "target" (and "class lables")
    model: model to predict
    tokenizer: tokenizer to tokenize the text
    device: device to run the model
    '''
    dataset = FinetuneDataset(texts = df["text"].values,
                              targets = df["target"].values,
                              tokenizer = tokenizer,
                              seq_len= tokenizer.model_max_length)
    
    data_loader = DataLoader(dataset, batch_size=16,
                            shuffle=False, num_workers=2)
    
    model.to(device)                                # Move model to the device.
    model.eval()                                    # Put model in evaluation mode.
    print('predicting...')
    predictions = []
    with torch.no_grad():                           # Disable gradient calculation.
        for batch in tqdm(data_loader):                   # Loop over all batches.   
            
            ids = batch["ids"].to(device, dtype=torch.long)
            masks = batch["masks"].to(device, dtype=torch.long)
            targets = batch["target"].to(device, dtype=torch.float)
            if mode == 'energy':
                # energy prediction
                outputs = model(ids, masks).squeeze(-1)
            elif mode == 'embed':
                # embedding generation
                outputs = model(ids, masks)['pooler_output']
            elif mode == 'attn':
                # attention generation
                outputs = model(ids, masks, output_attentions=True)['attentions'][0]
                return outputs
            else:
                raise ValueError('Mode not supported!')
            outputs = outputs.detach().cpu().numpy().tolist()
            predictions.extend(outputs)

    return predictions


if __name__ == '__main__':
    # ========================== INPUT ============================
    ckpt_dir = "checkpoint/finetune/base-ft_0623_0038" #pretrain/pt_0625_2026"
    ckpt_name = ckpt_dir.split('/')[-1]
    # =============================================================
    import argparse 
    parser = argparse.ArgumentParser(description='Set the running mode')
    parser.add_argument('--target', choices=['energy', 'embed', 'attn'], default='energy',
                    help='Prediction target (default: energy)')
    parser.add_argument('--base', action='store_true', help='Pretrain with base model') 
    args = parser.parse_args()
    pred = args.target
    if pred == 'energy':
        print('--------------- Energy prediction mode! ---------------')
    elif pred == 'embed':
        print('--------------- Embedding generation mode! ---------------')
    elif pred == 'attn':
        print('--------------- Attention generation mode! ---------------')
        # raise warning message 
        print('Warning: This mode is not fully tested yet!')
    # ================== 0. Load Checkpoint and Configs ==================
    if args.base:
        tag = 'base'
        val_data_path = "data/df_val.pkl"
        if pred == 'energy':
            raise ValueError('Cannot predict energy with base model!')
    
    else:
        tag = ckpt_name
        # Load Training Config
        if 'pt' in ckpt_name:
            print('This is a pretraining checkpoint!')
            if pred == 'energy':
                raise ValueError('Cannot predict energy with base model!')
            train_config = os.path.join(ckpt_dir, "pt_config.yaml")

        elif 'ft' in ckpt_name:
            train_config = os.path.join(ckpt_dir, "ft_config.yaml")

        paths = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)['paths']
        val_data_path = paths["val_data"]
        tknz_path = paths["tknz"]
        head = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)['params']['model_head']
        # Load Model Config
        if 'base' in ckpt_name:
            roberta_config = RobertaConfig.from_pretrained('roberta-base')
        else:
            model_config = yaml.load(open(os.path.join(ckpt_dir, "roberta_config.yaml"), "r"), Loader=yaml.FullLoader)
            roberta_config = RobertaConfig.from_dict(model_config)
        # Load Checkpoint
        checkpoint = os.path.join(ckpt_dir, "checkpoint.pt")
    # Set Device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # for debugging

    # ================== 1. Load Model and Tokenizer ==================
    if args.base:
        backbone = RobertaModel.from_pretrained('roberta-base')
        max_len = backbone.embeddings.position_embeddings.num_embeddings-2
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_len=max_len)
        model = backbone
    else:
        backbone = RobertaModel.from_pretrained('roberta-base', config=roberta_config, ignore_mismatched_sizes=True)
        max_len = backbone.embeddings.position_embeddings.num_embeddings-2
        model = backbone_wrapper(backbone, head)
        # model.load_state_dict(torch.load(checkpoint))
        tokenizer = RobertaTokenizerFast.from_pretrained(tknz_path, max_len=max_len)
        if pred == 'energy':
            # To generate energy, we need to load the model with the head.
            # So, we need to load the model from the checkpoint on the whole model.
            model = checkpoint_loader(model, checkpoint, load_on_roberta=False)
        elif pred == 'embed' or pred == 'attn':
            # To generate embeddings/attention, we need to load the model without the head.
            # So, we need to load the model from the checkpoint only on the backbone.
            model = checkpoint_loader(backbone, checkpoint, load_on_roberta=True)
                                    
    # ================== 2. Load Data ==================                               
    df_val = pd.read_pickle(val_data_path)
    # df_val = df_val.sample(5000, random_state=17) # for debugging
    
    # ================== 3. Obtain Predictions ==================
    predictions = predict_fn(df_val, model, tokenizer, device, mode=pred)
    
    # save predictions as dictionary with id as key
    results = {}
    for i in range(len(df_val)):
        results[df_val.iloc[i]['id']] = predictions[i]
    
    save_dir = f"results/{pred}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"catbert_{pred}_{tag}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)   
import torch
import pandas as pd
from transformers import RobertaModel, RobertaTokenizerFast
from tqdm import tqdm
from model.dataset import FinetuneDataset 
from model.common import * 
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
    import argparse 
    parser = argparse.ArgumentParser(description='Set the running mode')
    parser.add_argument('--target', choices=['energy', 'embed', 'attn'], default='energy',
                    help='Prediction target (default: energy)')
    parser.add_argument('--base', action='store_true', help='Pretrain with roberta-base model') 
    parser.add_argument('--ckpt_dir', required=True, help='Path to checkpoint directory')
    parser.add_argument('--data_path', required=True, help='Path to data directory')
    args = parser.parse_args()
    
    ckpt_dir = args.ckpt_dir
    data_path = args.data_path
    ckpt_name = ckpt_dir.split('/')[-1]
    data = pd.read_pickle(data_path)
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
        # this represents a roberta-base encoder without the regression head
        # thus, it's not for energy prediction
        tag = 'base'
        if pred == 'energy':
            raise ValueError('Cannot predict energy with base model!')
    
    else:
        # this represents a roberta-base encoder with the regression head
        # the checkpoint should be loaded on the whole model
        tag = ckpt_name
        # Load Training Config
        train_config = os.path.join(ckpt_dir, "ft_config.yaml")
        head = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)['params']['model_head']
        # Load Checkpoint
        checkpoint = os.path.join(ckpt_dir, "checkpoint.pt")

    # Set Device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ================== 1. Load Model and Tokenizer ==================
    backbone = RobertaModel.from_pretrained('roberta-base')
    max_len = backbone.embeddings.position_embeddings.num_embeddings-2
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    tokenizer.model_max_length = max_len
    if pred == 'energy':
        # To generate energy, we need to load the model with the head.
        # So, we need to load the model from the checkpoint on the whole model.
        model = backbone_wrapper(backbone, head)
        checkpoint_loader(model, checkpoint, load_on_roberta=False)
    
    elif pred == 'embed' or pred == 'attn':
        # To generate embeddings/attention, we need to load the model without the head.
        # So, we need to load the model from the checkpoint only on the backbone.
        if not args.base:
            checkpoint_loader(backbone, checkpoint, load_on_roberta=True)
        model = backbone                        

    # ================== 2. Obtain Predictions ==================
    predictions = predict_fn(data, model, tokenizer, device, mode=pred)
    
    # save predictions as dictionary with id as key
    results = {}
    for i in range(len(data)):
        results[data.iloc[i]['id']] = predictions[i]
    
    save_dir = f"results/{pred}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"catbert_{tag}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)   
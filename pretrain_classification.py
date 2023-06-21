import pandas as pd
import torch
import yaml
import os
import shutil
from transformers import (RobertaConfig, RobertaTokenizer, 
                          RobertaModel, RobertaTokenizerFast)
from model.pretrain_utils import run_pretraining
from model.common import backbone_wrapper
from datetime import datetime
import argparse
parser = argparse.ArgumentParser(description='Set the running mode')
parser.add_argument('--debug', action='store_true', help='Enable debugging mode')
args = parser.parse_args()
# ==========================================================
# Set up paths for data, model, tokenizer, and results
# ==========================================================
pt_config_path = "./config/pt_config.yaml"
paths = yaml.load(open(pt_config_path, "r"), Loader=yaml.FullLoader)['paths']
train_data_path = paths["train_data"]
val_data_path = paths["val_data"]
rbt_config_path = paths["roberta_config"]
tknz_path = paths["tknz"]
ckpt_for_further_train = './checkpoint/pretrain/pt_0620_1427/checkpoint.pt'
# ==========================================================
if args.debug:
    print('Debugging mode enabled!')
# Load data
df_train = pd.read_pickle(train_data_path) #.sample(100, random_state=42)
df_val = pd.read_pickle(val_data_path) #.sample(100, random_state=42)
# for debugging
if args.debug:
    df_train = df_train.sample(1, random_state=42)
    df_val = df_val.sample(2, random_state=42)
print("Training data size: " + str(df_train.shape[0]))
print("Validation data size : " + str(df_val.shape[0]))

# Set hyperparameters
params = yaml.load(open(pt_config_path, "r"), Loader=yaml.FullLoader)['params']
model_config = yaml.load(open(rbt_config_path, 'r'), Loader=yaml.FullLoader) #['roberta_config']
roberta_config = RobertaConfig.from_dict(model_config)

# Load pre-trained backbone model
backbone = RobertaModel.from_pretrained('roberta-base', config=roberta_config, ignore_mismatched_sizes=True)
model = backbone_wrapper(backbone, params['model_head'])
# import pdb; pdb.set_trace()
# if start training from pretrained header
# model.load_state_dict(torch.load(ckpt_for_further_train)) #torch.load(ckpt_path)
# Load pre-trained tokenizer
max_len = backbone.embeddings.position_embeddings.num_embeddings
tokenizer = RobertaTokenizerFast.from_pretrained(tknz_path, max_len=max_len)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.debug:
    device = torch.device("cpu") # for debugging
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')    
    
# run training
run_name = 'pt'+datetime.now().strftime("_%m%d_%H%M") #+'_on_0620_1427'
if args.debug:
    run_name = "debugging"
run_pretraining(df_train, df_val, params, model, tokenizer, device, run_name=run_name)

# save config files for reference
shutil.copy(pt_config_path, os.path.join(f"./checkpoint/pretrain/{run_name}", "pt_config.yaml"))
shutil.copy(rbt_config_path, os.path.join(f"./checkpoint/pretrain/{run_name}", "roberta_config.yaml"))
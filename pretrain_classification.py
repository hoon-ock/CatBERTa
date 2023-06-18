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

# ==========================================================
# Set up paths for data, model, tokenizer, and results
# ==========================================================
pt_config_path = "./config/pt_config.yaml"
paths = yaml.load(open(pt_config_path, "r"), Loader=yaml.FullLoader)['paths']
train_data_path = paths["train_data"]
val_data_path = paths["val_data"]
rbt_config_path = paths["roberta_config"]
tknz_path = paths["tknz"]
# ==========================================================

# Load data
df_train = pd.read_pickle(train_data_path)
df_val = pd.read_pickle(val_data_path)
# for debugging
df_train = df_train.sample(500, random_state=42)
df_val = df_val.sample(20, random_state=42)
print("Training data size: " + str(df_train.shape[0]))
print("Validation data size : " + str(df_val.shape[0]))

# Set hyperparameters
params = yaml.load(open(pt_config_path, "r"), Loader=yaml.FullLoader)['params']
config = yaml.load(open(rbt_config_path, 'r'), Loader=yaml.FullLoader)['roberta_config']
roberta_config = RobertaConfig.from_dict(config)

# Load pre-trained backbone model
backbone = RobertaModel.from_pretrained('roberta-base', config=roberta_config, ignore_mismatched_sizes=True)

model = backbone_wrapper(backbone, params['model_head'])
# if start training from pretrained header
# model = model.load_state_dict(torch.load(ckpt_path2)) #torch.load(ckpt_path)

# Load pre-trained tokenizer
max_len = backbone.embeddings.position_embeddings.num_embeddings
tokenizer = RobertaTokenizerFast.from_pretrained(tknz_path, max_len=max_len)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu") # for debugging
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')    
    print('empty cache!')
    torch.cuda.empty_cache()
# run training
run_name = "debugging" #'pt'+datetime.now().strftime("_%m%d_%H%M")
run_pretraining(df_train, df_val, params, model, tokenizer, device, run_name=run_name)

# save config files for reference
shutil.copy(pt_config_path, os.path.join(f"./checkpoint/pretrain/{run_name}", "pt_config.yaml"))
shutil.copy(rbt_config_path, os.path.join(f"./checkpoint/pretrain/{run_name}", "roberta_config.yaml"))
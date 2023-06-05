import pandas as pd
import torch
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, RobertaTokenizerFast
from model.train_utils import run_training
from model.dataset import stratified_kfold
import json


# ==========================================================
# Set up paths for data, model, tokenizer, and results
# ==========================================================
train_data_path = "./data/df_is2re_100k.pkl"
val_data_path = "./data/df_is2re_val_25k.pkl"
config_path = "./config/roberta_config.json"
ckpt_path = "./checkpoint/pretrain/len768_ep10_bs16_0602_1820/"
#"./checkpoint/pretrain/len768_ep10_bs16_0530_1554/"
#"./checkpoint/pretrain/len768_ep5_bs16_0529_1555.pt"
tknz_path = "./tokenizer"
custom_save_path = "./results/dummy/"
# ==========================================================

# Load data
df_train = pd.read_pickle(train_data_path)
df_val = pd.read_pickle(val_data_path)
#df = df.iloc[:20000] # for code testing
# for debugging
# df_train = df_train.iloc[:500]
# df_val = df_val.iloc[:200]
print("Training data size: " + str(df_train.shape[0]))
print("Validation data size : " + str(df_val.shape[0]))

# Set hyperparameters
with open(config_path, "r") as f:
    loaded_dict = json.load(f)
params = loaded_dict['finetune_params']


# Load (pre-trained) model
# config = RobertaConfig.from_dict(loaded_dict['roberta_config'])
# model = RobertaModel.from_pretrained('roberta-base')
                                    #   config=config,
                                    #  ignore_mismatched_sizes=True)
model = RobertaModel.from_pretrained(ckpt_path) #, config=config)
max_len = model.embeddings.position_embeddings.num_embeddings
# Load (pre-trained) tokenizer
# tokenizer =  RobertaTokenizer.from_pretrained("roberta-base")
tokenizer = RobertaTokenizerFast.from_pretrained(tknz_path, 
                                                 max_len=max_len)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu") # for debugging
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')    

# run training
run_training(df_train, df_val, params, model,tokenizer,device, 
             run_name='100k_own-tknz-mdl_attnhd') 
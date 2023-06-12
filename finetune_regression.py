import pandas as pd
import torch
from transformers import (RobertaConfig, RobertaTokenizer, 
                          RobertaForMaskedLM, RobertaModel, 
                          RobertaTokenizerFast)
from model.train_utils import run_training
from model.dataset import stratified_kfold
from model.regressors import (MyModel, MyModel2, 
                              MyModel_MLP, MyModel_AttnHead, 
                              MyModel_ConcatLast4Layers)
import json
from sklearn.model_selection import train_test_split

# ==========================================================
# Set up paths for data, model, tokenizer, and results
# ==========================================================
train_data_path = "./data/df_is2re_100k_new.pkl"
val_data_path = "./data/df_is2re_val_25k_new.pkl"
config_path = "./config/roberta_config.json"
ckpt_path = "./checkpoint/pretrain/len768_ep10_bs16_0602_1820/"
# len512_ep10_bs16_0605_1329, len768_ep10_bs16_0602_1820
tknz_path = "./tokenizer"
# ==========================================================

# Load data
df_train = pd.read_pickle(train_data_path)
df_val = pd.read_pickle(val_data_path)
# for debugging
# df_train = df_train.sample(1000, random_state=42)
# df_val = df_val.sample(100, random_state=42)
test_size = 0.2
df_train, df_val = train_test_split(df_train, test_size=test_size, random_state=42)
print("Training data size: " + str(df_train.shape[0]))
print("Validation data size : " + str(df_val.shape[0]))

# Set hyperparameters
with open(config_path, "r") as f:
    loaded_dict = json.load(f)
params = loaded_dict['finetune_params']


# Load (pre-trained) model
# config = RobertaConfig.from_dict(loaded_dict['roberta_config'])
# model = RobertaModel.from_pretrained('roberta-base', 
#                                      config=config, 
#                                      ignore_mismatched_sizes=True)
    
model = RobertaModel.from_pretrained(ckpt_path)
max_len = model.embeddings.position_embeddings.num_embeddings

# Load (pre-trained) tokenizer
# tokenizer =  RobertaTokenizer.from_pretrained("roberta-base")
tokenizer = RobertaTokenizerFast.from_pretrained(tknz_path, max_len=max_len)
#import pdb; pdb.set_trace()

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
run_training(df_train, df_val, params, model,tokenizer, device, 
             run_name='n80k_mlp_len768_lr5e-7_L2-wd001_sch-const')
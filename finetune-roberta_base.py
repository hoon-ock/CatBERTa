import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
from model.train_utils import run_training
from model.dataset import stratified_kfold

# check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')    

# Load data
df = pd.read_pickle('/home/jovyan/CATFORMER/data/df_train.pkl')
df = df.iloc[:5000] # for code testing
print("Training dataset shape : " + str(df.shape))
df = stratified_kfold(df, n_splits=5)

# Load tokenizer
tokenizer =  RobertaTokenizer.from_pretrained("roberta-base")

# Load model

# run training
run_training(df, tokenizer, model_head="pooler") 
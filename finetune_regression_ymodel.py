import pandas as pd
import torch, yaml, os, shutil, copy
from transformers import (RobertaConfig, RobertaTokenizer, 
                          RobertaModel, RobertaTokenizerFast)
from model.ymodel_finetune_utils import run_finetuning
from model.common import backbone_wrapper, checkpoint_loader
from model.regressors import YModel
from datetime import datetime


import argparse 
parser = argparse.ArgumentParser(description='Set the running mode')
parser.add_argument('--debug', action='store_true', help='Enable debugging mode')
parser.add_argument('--alpha', type=float, default=0.5, help='Set the alpha value for YModel')
args = parser.parse_args()
if args.debug:
    print('Debugging mode enabled!')
alpha = args.alpha

#    #################################################
# temporary function
def section_text_integrator(data, col_list):
    df = data.copy()
    df['text'] = df[col_list].apply(lambda x: ' '.join(x), axis=1)
    return df
# ============== 0. Read finetune config file ======================
ft_config_path = "config/ft_config.yaml"
paths = yaml.load(open(ft_config_path, "r"), Loader=yaml.FullLoader)['paths']
train_data_path = paths["train_data"] 
val_data_path = paths["val_data"] 
pt_ckpt_path = paths["pt_ckpt"] 
tknz_path = paths["tknz"]
# ckpt_for_further_train = 'checkpoint/finetune/ft_0619_0223/checkpoint.pt'
print("This model is based on: ", pt_ckpt_path.split('/')[-1])
# ================= 1. Load data ======================
df_train = pd.read_pickle(train_data_path)
df_val = pd.read_pickle(val_data_path)
##########################################################
# temporary code
# integrate the text
df_train = section_text_integrator(df_train, ['system'])
df_val = section_text_integrator(df_val, ['system'])
# breakpoint()
##########################################################
if args.debug:
    df_train = df_train.sample(8, random_state=42)
    df_val = df_val.sample(4, random_state=42)
print("Training data size: " + str(df_train.shape[0]))
print("Validation data size : " + str(df_val.shape[0]))

# ================= 2. Load model ======================
params = yaml.load(open(ft_config_path, "r"), Loader=yaml.FullLoader)['params']
# Load pre-trained backbone model
backbone = RobertaModel.from_pretrained('roberta-base')
# need to change the model head part!!!!!
pretrained_model = backbone_wrapper(backbone, 'pooler') 
checkpoint_loader(pretrained_model, os.path.join(pt_ckpt_path, 'checkpoint.pt'), load_on_roberta=False)
print("Word Embedding: ", backbone.embeddings.word_embeddings)

# Load submodel for finetuning
sub_model = RobertaModel.from_pretrained('roberta-base')

# Load YModel
model = YModel(pretrained_model, sub_model)

# ================= 3. Load tokenizer ======================
max_len = backbone.embeddings.position_embeddings.num_embeddings-2
tokenizer = RobertaTokenizerFast.from_pretrained(tknz_path, max_len=max_len)
print("Max length: ", tokenizer.model_max_length)

# ================= 4. Set device ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.debug:
    device = torch.device('cpu')
print("Device: " + str(device))
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')        

# ================= 5. Run finetuning ======================
run_name = f'ymodel-{alpha}-ft'+datetime.now().strftime("_%m%d_%H%M")
if args.debug:
    run_name = 'debug'+datetime.now().strftime("_%m%d_%H%M")
print("Run name: ", run_name)
# breakpoint()
save_dir = os.path.join(f"checkpoint/finetune/{run_name}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# save config files for reference
shutil.copy(ft_config_path, os.path.join(save_dir, "ft_config.yaml"))

# run finetuning
run_finetuning(df_train, df_val, params, alpha, model, tokenizer, device, run_name= run_name)
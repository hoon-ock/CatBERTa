import pandas as pd
import torch, yaml, os, shutil, copy
from transformers import (BertTokenizerFast, BertModel, BertConfig,
                          AutoTokenizer, AutoModel)
from model.finetune_utils import run_finetuning
from model.common import backbone_wrapper, checkpoint_loader
from datetime import datetime
import argparse 
parser = argparse.ArgumentParser(description='Set the running mode')
parser.add_argument('--debug', action='store_true', help='Enable debugging mode')
args = parser.parse_args()
if args.debug:
    print('Debugging mode enabled!')

def count_parameters(model):
    """
    Count the total number of parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_layers(model, freeze_layers):
    for name, param in model.named_parameters():
        if freeze_layers == 'all':
            param.requires_grad = False
        else:
            if name in freeze_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
    

# ============== 0. Read pretrain config file ======================
ft_config_path = "config/ft_config.yaml"
paths = yaml.load(open(ft_config_path, "r"), Loader=yaml.FullLoader)['paths']
train_data_path = paths["train_data"] 
val_data_path = paths["val_data"] 
pt_ckpt_path = paths["pt_ckpt"] 
tknz_path = paths["tknz"]
print("This model is based on: ", pt_ckpt_path.split('/')[-1])

# ================= 1. Load data ======================
df_train = pd.read_pickle(train_data_path)
df_val = pd.read_pickle(val_data_path)
if args.debug:
    df_train = df_train.sample(4, random_state=42)
    df_val = df_val.sample(4, random_state=42)
print("Training data size: " + str(df_train.shape[0]))
print("Validation data size : " + str(df_val.shape[0]))

# ================= 2. Load model and tokenizer ======================
params = yaml.load(open(ft_config_path, "r"), Loader=yaml.FullLoader)['params']
# Load pre-trained backbone model
if pt_ckpt_path == 'm3rg-iitd/matscibert':
    backbone = BertModel.from_pretrained("m3rg-iitd/matscibert")
    #backbone = BertModel.from_pretrained('matbert-base-cased')
    max_len = backbone.embeddings.position_embeddings.num_embeddings
    tokenizer = BertTokenizerFast.from_pretrained("m3rg-iitd/matscibert", model_max_length=max_len)
    #tokenizer = BertTokenizerFast.from_pretrained('matbert-base-cased', model_max_length=max_len, do_lower_case=True)

else:
    print("----------------------------------------------------")
    print("under construction...")
    print("----------------------------------------------------")
    exit()

# wrap with a regression head
#freeze_layers(backbone, freeze_layers='all')
model = backbone_wrapper(backbone, params['model_head'])
print("Max length: ", tokenizer.model_max_length)
# breakpoint()

# ================= 3. Set device ======================
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

# ================= 4. Run pretraining ======================
run_name = 'msb-ft'+datetime.now().strftime("_%m%d_%H%M")
if args.debug:
    run_name = 'debug'+datetime.now().strftime("_%m%d_%H%M")
if pt_ckpt_path == 'roberta-base':
    run_name = 'base-'+run_name
print("Run name: ", run_name)

save_dir = os.path.join(f"checkpoint/finetune/{run_name}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# save config files for reference
shutil.copy(ft_config_path, os.path.join(save_dir, "ft_config.yaml"))
# shutil.copy(os.path.join(pt_ckpt_path, 'roberta_config.yaml'), os.path.join(save_dir, "roberta_config.yaml"))

# run finetuning
run_finetuning(df_train, df_val, params, model, tokenizer, device, run_name=run_name)
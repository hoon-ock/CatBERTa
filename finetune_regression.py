import pandas as pd
import torch, yaml, os, shutil
from transformers import RobertaModel, RobertaTokenizerFast
from model.finetune_utils import run_finetuning
from model.common import * 
from datetime import datetime
import argparse 
parser = argparse.ArgumentParser(description='Set the running mode')
parser.add_argument('--debug', action='store_true', help='Enable debugging mode')
args = parser.parse_args()
if args.debug:
    print('Debugging mode enabled!')


# ============== 0. Read finetune config file ======================
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

# ================= 2. Load model ======================
params = yaml.load(open(ft_config_path, "r"), Loader=yaml.FullLoader)['params']
backbone = RobertaModel.from_pretrained('roberta-base')
# wrap with a regression head
model = backbone_wrapper(backbone, params['model_head'])
# if start training from pretrained header
# ckpt_for_further_train = torch.load(os.path.join(pt_ckpt_path, 'checkpoint.pt'))
# model.load_state_dict(ckpt_for_further_train) #torch.load(ckpt_path)

# ================= 3. Load tokenizer ======================
max_len = backbone.embeddings.position_embeddings.num_embeddings-2
tokenizer = RobertaTokenizerFast.from_pretrained(tknz_path)
tokenizer.model_max_length = max_len
print("Max length: ", tokenizer.model_max_length)

# ================= 4. Set device ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.debug:
    device = torch.device('cpu')
print("Device: " + str(device))

# check memory usage
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')        

# ================= 5. Run finetuning ======================
run_name = 'ft'+datetime.now().strftime("_%m%d_%H%M")
if args.debug:
    run_name = 'debug'+datetime.now().strftime("_%m%d_%H%M")
print("Run name: ", run_name)

save_dir = os.path.join(f"checkpoint/finetune/{run_name}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# save config files for future reference
shutil.copy(ft_config_path, os.path.join(save_dir, "ft_config.yaml"))

# run finetuning
run_finetuning(df_train, df_val, params, model, tokenizer, device, run_name= run_name)
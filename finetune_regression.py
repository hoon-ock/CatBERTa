import pandas as pd
import torch, yaml, os, shutil, copy
from transformers import (RobertaConfig, RobertaTokenizer, 
                          RobertaModel, RobertaTokenizerFast)
from model.finetune_utils import run_finetuning
from model.common import backbone_wrapper, checkpoint_loader
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import argparse 
parser = argparse.ArgumentParser(description='Set the running mode')
parser.add_argument('--debug', action='store_true', help='Enable debugging mode')
#parser.add_argument('--base', action='store_true', help='Finetune with base model') 
args = parser.parse_args()
if args.debug:
    print('Debugging mode enabled!')
# if args.base:
#     # bypassing the pretraining step
#    #################################################
# temporary function
def section_text_integrator(data, col_list):
    df = data.copy()
    df['text'] = df[col_list].apply(lambda x: ' '.join(x), axis=1)
    return df

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

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

# df_train = section_text_integrator(df_train, ['string2'])
# df_val = pd.read_pickle('/home/jovyan/shared-scratch/jhoon/CATBERT/is2re/val/500_type1.pkl')
# df_val = section_text_integrator(df_val, ['string2'])
df_train['text'] = df_train['text'].apply(lambda x: x.replace('*', ''))
# df_train['text'] = df_train['text'].apply(lambda x: f"{x.split('</s>')[0]}</s>{x.split('</s>')[-1]}")
# df_train['text'] = df_train['text'].apply(lambda x: f"{x.split('</s>')[0]}</s>{x.split('</s>')[-1]}")
# df_train['text'] = df_train['text'].apply(lambda x: x.replace(' atop', ''))
# df_train['text'] = df_train['text'].apply(lambda x: x.replace(' hollow', ''))
# df_train['text'] = df_train['text'].apply(lambda x: x.replace(' bridge', ''))

df_val['text'] = df_val['text'].apply(lambda x: x.replace('*', ''))
# df_val['text'] = df_val['text'].apply(lambda x: f"{x.split('</s>')[0]}</s>{x.split('</s>')[-1]}")
# df_val['text'] = df_val['text'].apply(lambda x: x.replace(' atop', ''))
# df_val['text'] = df_val['text'].apply(lambda x: x.replace(' bridge', ''))
# df_val['text'] = df_val['text'].apply(lambda x: x.replace(' hollow', ''))
# breakpoint()
# print(df_train.describe())
# df_train_subset = pd.read_pickle('data/df_train_config2.pkl')
# df_train = df_train[df_train['id'].isin(df_train_subset['id'].values)]
# df_val_subset = pd.read_pickle('data/df_val_config2.pkl')
# df_val = df_val[df_val['id'].isin(df_val_subset['id'].values)]

# df_train = df_train.sample(50000)
# df_val = df_val.sample(1000)
# split with </s>, then rejoin with </s>
# removing the second split section by </s>
# df_train['text'] = ['</s>'.join(parts[:2] + [parts[-1].replace(' ', '')]) for parts in (text.split('</s>') for text in df_train['text'].values)]
# df_val['text'] = ['</s>'.join(parts[:2] + [parts[-1].replace(' ', '')]) for parts in (text.split('</s>') for text in df_val['text'].values)]

# breakpoint()
# scale the labels
# energy_scaler = StandardScaler()
# energy_scaler = MinMaxScaler()
# energy_scaler = RobustScaler()
# energy_scaler.fit(df_train['target'].values.reshape(-1,1))
# train_labels = energy_scaler.transform(df_train['target'].values.reshape(-1,1))
# val_labels = energy_scaler.transform(df_val['target'].values.reshape(-1,1))
# df_train['target'] = train_labels
# df_val['target'] = val_labels
# df_train = df_train.sample(10000, random_state=42)
# df_val = df_val.sample(1000, random_state=42)

# breakpoint()
##########################################################
if args.debug:
    df_train = df_train.sample(4, random_state=42)
    df_val = df_val.sample(4, random_state=42)
print("Training data size: " + str(df_train.shape[0]))
print("Validation data size : " + str(df_val.shape[0]))

# ================= 2. Load model ======================
params = yaml.load(open(ft_config_path, "r"), Loader=yaml.FullLoader)['params']
# Load pre-trained backbone model
# if pt_ckpt_path == 'roberta-base':
#     backbone = RobertaModel.from_pretrained('roberta-base')
# else:
#     config = yaml.load(open(os.path.join(pt_ckpt_path, 'roberta_config.yaml'), 'r'), Loader=yaml.FullLoader)
#     roberta_config = RobertaConfig.from_dict(config)
#     backbone = RobertaModel.from_pretrained('roberta-base', config=roberta_config, ignore_mismatched_sizes=True)
#     backbone_base = copy.deepcopy(backbone)
#     checkpoint_loader(backbone, os.path.join(pt_ckpt_path, 'checkpoint.pt'), load_on_roberta=True)
#     ############### checkpoint loading sanity check! ###############
#     # backbone_base = copy.deepcopy(backbone)
#     base_emb = backbone_base.embeddings.word_embeddings.weight
#     backbone_emb = backbone.embeddings.word_embeddings.weight
#     if torch.equal(base_emb, backbone_emb):
#         print("Checkpoint loading failed!")
#         raise ValueError
#     del backbone_base, base_emb, backbone_emb
#     ###############################################################

# roberta_config = RobertaConfig.from_dict({
#                                         'hidden_size': 768,
#                                         'num_attention_heads': 12,
#                                         'num_hidden_layers': 16,
#                                         'type_vocab_size': 1,
#                                         'vocab_size': 50265,
#                                         'max_position_embeddings': 258,
#                                         })
# backbone = RobertaModel.from_pretrained('roberta-base', config=roberta_config, ignore_mismatched_sizes=True)
# breakpoint()
# roberta_config = RobertaConfig.from_pretrained('roberta-base')
# roberta_config.vocab_size = 2118
# roberta_config.max_position_embeddings = 258
# backbone = RobertaModel.from_pretrained('roberta-base', config=roberta_config, ignore_mismatched_sizes=True)


# tokenizer = RobertaTokenizerFast.from_pretrained(tknz_path)
# tokenizer.model_max_length = 256 #model.roberta.embeddings.position_embeddings.num_embeddings-2
# roberta_config = RobertaConfig.from_pretrained('roberta-base')
# roberta_config.vocab_size = tokenizer.vocab_size
# roberta_config.max_position_embeddings = 258
# backbone = RobertaModel.from_pretrained('roberta-base', config=roberta_config, ignore_mismatched_sizes=True)

backbone = RobertaModel.from_pretrained('roberta-base')

# backbone = RobertaModel.from_pretrained('roberta-base')
# checkpoint_loader(backbone, os.path.join(pt_ckpt_path, 'checkpoint.pt'), load_on_roberta=True)
# breakpoint()
# wrap with a regression head
model = backbone_wrapper(backbone, params['model_head'])
print("Word Embedding: ", backbone.embeddings.word_embeddings)
breakpoint()
# if start training from pretrained header
# ckpt_for_further_train = torch.load(os.path.join(pt_ckpt_path, 'checkpoint.pt'))
# model.load_state_dict(ckpt_for_further_train) #torch.load(ckpt_path)
# breakpoint()
# ================= 3. Load tokenizer ======================
max_len = backbone.embeddings.position_embeddings.num_embeddings-2
tokenizer = RobertaTokenizerFast.from_pretrained(tknz_path)
tokenizer.model_max_length = max_len
print("Max length: ", tokenizer.model_max_length)
# ================= 4. Set device ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # for debugging
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
run_name = 'des1-rebase-ft'+datetime.now().strftime("_%m%d_%H%M")
if args.debug:
    run_name = 'debug'+datetime.now().strftime("_%m%d_%H%M")
# if pt_ckpt_path == 'roberta-base':
#     run_name = 'base-'+run_name
print("Run name: ", run_name)
# breakpoint()
save_dir = os.path.join(f"checkpoint/finetune/{run_name}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# save config files for reference
shutil.copy(ft_config_path, os.path.join(save_dir, "ft_config.yaml"))
#if pt_ckpt_path != 'roberta-base':
#    shutil.copy(os.path.join(pt_ckpt_path, 'roberta_config.yaml'), os.path.join(save_dir, "roberta_config.yaml"))

# run finetuning
run_finetuning(df_train, df_val, params, model, tokenizer, device, run_name= run_name)
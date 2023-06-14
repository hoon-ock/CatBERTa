import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class MyDataset(torch.utils.data.Dataset):
   
    def __init__(self, texts, targets, tokenizer, seq_len=500): 
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.seq_len = seq_len # 400 was set as the default originally  
        # self.mean = mean 
        # self.std = std 
        
    def __len__(self):
        """Returns the length of dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        For a given item index, return a dictionary of encoded information        
        """        
        text = str(self.texts[idx]) 
        
        tokenized = self.tokenizer(
            text,            
            max_length = self.seq_len,                                
            padding = "max_length",     # Pad to the specified max_length. 
            truncation = True,          # Truncate to the specified max_length. 
            add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
            return_attention_mask = True            
        )     
        # Normalization of target values
        # target = (self.targets[idx] - self.mean) / self.std

        return {"ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
                "masks": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
                "target": torch.tensor(self.targets[idx], dtype=torch.float)
               }


class PretrainDataset(torch.utils.data.Dataset):
   
    def __init__(self, texts, ads_size, ads_class, bulk_class, tokenizer, seq_len=500): 
        self.texts = texts
        self.ads_size = ads_size
        self.ads_class = ads_class
        self.bulk_class = bulk_class
        self.tokenizer = tokenizer
        self.seq_len = seq_len 
        
    def __len__(self):
        """Returns the length of dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        For a given item index, return a dictionary of encoded information        
        """        
        text = str(self.texts[idx]) 
        
        tokenized = self.tokenizer(
            text,            
            max_length = self.seq_len,                                
            padding = "max_length",     # Pad to the specified max_length. 
            truncation = True,          # Truncate to the specified max_length. 
            add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
            return_attention_mask = True            
        )     


        return {"ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
                "masks": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
                "ads_size": torch.tensor(self.ads_size[idx], dtype=torch.int),
                "ads_class": torch.tensor(self.ads_class[idx], dtype=torch.int),
                "bulk_class": torch.tensor(self.bulk_class[idx], dtype=torch.int)
               }


def stratified_kfold(df, n_splits=5):
    """
    Returns a dataframe with a new column "skfold" indicating the fold number for each record.
    """
    df["skfold"] = -99
    df = df.sample(frac=1).reset_index(drop=True)
    bins = int(np.floor(1 + np.log2(len(df))))
    df.loc[:, "bins"] = pd.cut(df["target"], bins=bins, labels=False)
    skf = StratifiedKFold(n_splits = n_splits)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df["bins"].values)):
        df.loc[val_idx, "skfold"] = fold
    return df
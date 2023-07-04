import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class FinetuneDataset(torch.utils.data.Dataset):
   
    def __init__(self, texts, targets, tokenizer, seq_len=512): 
        self.texts = texts
        self.targets = targets
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
                "target": torch.tensor(self.targets[idx], dtype=torch.float)
               }


class ClassDataset(torch.utils.data.Dataset):
   
    def __init__(self, texts, ads_size, ads_class, bulk_class, tokenizer, seq_len=512): 
        self.texts = texts
        self.ads_size = ads_size
        self.ads_class = ads_class
        self.bulk_class = bulk_class
        self.tokenizer = tokenizer
        self.seq_len = seq_len 
        self.n_ads_size = 3 #max(ads_size) +1
        self.n_ads_class = 5 #max(ads_class) + 1
        self.n_bulk_class = 4 #max(bulk_class) + 1
        
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
                "labels":{
                          "ads_size": torch.tensor(np.eye(self.n_ads_size)[self.ads_size[idx]]),
                          "ads_class": torch.tensor(np.eye(self.n_ads_class)[self.ads_class[idx]]),
                          "bulk_class": torch.tensor(np.eye(self.n_bulk_class)[self.bulk_class[idx]])
                }
               }

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, texts_sys, texts_ads, texts_bulk, targets, tokenizer, seq_len): 
        '''
        texts: text of each section (e.g. "system", "adsorbate", "bulk")
               system text includes general system description and configuration description
        targets: relaxed energy of the system
        tokenizer: tokenizer (roberta-base or own tokenizer)
        seq_len: maximum length of the input sequence (e.g. 256, 512)
        '''
        self.texts_sys = texts_sys 
        self.texts_ads = texts_ads
        self.texts_bulk = texts_bulk
        self.targets = targets
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sections = ['system', 'adsorbate', 'bulk'] #list(texts.keys())

        
    def __len__(self):
        """Returns the length of dataset."""
        # raise error if the length of all sections are not the same
        assert len(self.texts_sys) == len(self.texts_ads) == len(self.texts_bulk) == len(self.targets), \
            'The length of all sections must be the same.'
        return len(self.texts_sys)
    
    def __getitem__(self, idx):
        """
        For a given item index, return a dictionary of encoded information   
        return_dict:
        e.g. {'ids': {'system': tensor([  0,  10,  14,  16]),
                      'adsorbate': tensor([  0,  10,  14,  16]),
                      'bulk': tensor([  0,  10,  14,  16])}
              'masks': {'system': tensor([1, 1, 1, 1]),
                        'adsorbate': tensor([1, 1, 1, 1]),  
                        'bulk': tensor([1, 1, 1, 1])},
             'target': tensor(2.0123)}
        """   
        return_dict = {
                        'ids': {}, 
                        'masks': {}, 
                        'target': self.targets[idx]
                        }
        for section, text in zip(self.sections, [self.texts_sys, self.texts_ads, self.texts_bulk]):
            section_text = str(text[idx])
            tokenized = self.tokenizer(
                section_text,            
                max_length = self.seq_len,                                
                padding = "max_length",     # Pad to the specified max_length. 
                truncation = True,          # Truncate to the specified max_length. 
                add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
                return_attention_mask = True            
            )
            return_dict['ids'][section] = torch.tensor(tokenized["input_ids"], dtype=torch.long)
            return_dict['masks'][section] = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
        return return_dict




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
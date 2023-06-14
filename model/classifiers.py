import torch
from torch import Tensor
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class BinaryClassifier(nn.Module):
            
    def __init__(self, model):        
        super().__init__() 
        self.roberta_model = model 
        self.classifier = nn.Sequential(
                        nn.Linear(768, 768),
                        nn.Dropout(0.1),
                        nn.Linear(768, 2), # 2: number of classes
                        nn.ReLU()
        )     

    def forward(self, input_ids, attention_mask):        
        raw_output = self.roberta_model(input_ids, attention_mask, return_dict=True)        
        pooler = raw_output["pooler_output"]    # Shape is [batch_size, 768]
        output = self.classifier(pooler)         # Shape is [batch_size, 1]
        return output 
    

class MultiLabelClassifier(nn.Module):
            
    def __init__(self, model, n_ads_size=3, n_ads_class=5, n_bulk_class=4):        
        super().__init__() 
        self.roberta_model = model 
        self.ads_size = nn.Sequential(
                        nn.Linear(768, 768),
                        nn.Dropout(0.1),
                        nn.Linear(768, n_ads_size), 
        )

        self.ads_class = nn.Sequential(
                        nn.Linear(768, 768),
                        nn.Dropout(0.1),
                        nn.Linear(768, n_ads_class),
        )

        self.bulk_class = nn.Sequential(
                        nn.Linear(768, 768),
                        nn.Dropout(0.1),
                        nn.Linear(768, n_bulk_class),
        )



    def forward(self, input_ids, attention_mask):        
        raw_output = self.roberta_model(input_ids, attention_mask, return_dict=True)        
        pooler = raw_output["pooler_output"]    # Shape is [batch_size, 768]
        output = {'ads_size': self.ads_size(pooler), 
                  'ads_class': self.ads_class(pooler), 
                  'bulk_class': self.bulk_class(pooler) 
                  }
        return output            
    
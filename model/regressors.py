import torch
from torch import Tensor
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class MyModel(nn.Module):
            
    def __init__(self):        
        super().__init__() 
        self.roberta_config = RobertaConfig(vocab_size=30_522,
                                            max_position_embeddings=514,
                                            hidden_size=768,
                                            num_attention_heads=12,
                                            num_hidden_layers=6,
                                            type_vocab_size=1,)
        self.pretrained_path = '/home/jovyan/CATBERT/checkpoint/pretrain/data55092_ep10_bs32_20230527_001513.pt'
        self.roberta_model = RobertaModel.from_pretrained(self.pretrained_path, config=self.roberta_config)                                    
        #self.roberta_model = RobertaModel.from_pretrained('roberta-base', config=self.roberta_config) #, ignore_mismatched_sizes=True)      
        self.regressor = nn.Linear(768, 1)     

    def forward(self, input_ids, attention_mask):        
        raw_output = self.roberta_model(input_ids, attention_mask, return_dict=True)        
        pooler = raw_output["pooler_output"]    # Shape is [batch_size, 768]
        output = self.regressor(pooler)         # Shape is [batch_size, 1]
        return output 

class regressionHead(nn.Module):
    def __init__(self, d_embedding: int =768):
        super().__init__()
        self.layer1 = nn.Linear(d_embedding, d_embedding//2)
        self.layer2 = nn.Linear(d_embedding//2, d_embedding//4)
        self.layer3 = nn.Linear(d_embedding//4, d_embedding//8)
        self.layer4 = nn.Linear(d_embedding//8, 1)
        self.relu=nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.layer4(x)

class MyModel_MLP(nn.Module):
            
    def __init__(self):        
        super().__init__() 
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')       
        self.regressor = regressionHead() #nn.Linear(768, 1)     

    def forward(self, input_ids, attention_mask):        
        raw_output = self.roberta_model(input_ids, attention_mask, return_dict=True)        
        pooler = raw_output["pooler_output"]    # Shape is [batch_size, 768]
        output = self.regressor(pooler)         # Shape is [batch_size, 1]
        return output 


class AttentionHead(nn.Module):
    
    def __init__(self, input_dim=768, hidden_dim=512):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, last_hidden_state):
        """
        Note:
        "last_hidden_state" shape is [batch_size, seq_len, 768].
        The "weights" produced from softmax will add up to 1 across all tokens in each record.
        """        
        linear1_output = self.linear1(last_hidden_state)  # Shape is [batch_size, seq_len, 512]  
        activation = torch.tanh(linear1_output)           # Shape is [batch_size, seq_len, 512]        
        score = self.linear2(activation)                  # Shape is [batch_size, seq_len, 1]        
        weights = torch.softmax(score, dim=1)             # Shape is [batch_size, seq_len, 1]              
        result = torch.sum(weights * last_hidden_state, dim=1) # Shape is [batch_size, 768]          
        return result
    

class MyModel_AttnHead(nn.Module):
            
    def __init__(self):        
        super().__init__() 
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.attn_head = AttentionHead(768, 512)       
        self.regressor = nn.Linear(768, 1)     
    
    def forward(self, input_ids, attention_mask):       
        raw_output = self.roberta_model(input_ids, attention_mask, return_dict=True)        
        last_hidden_state = raw_output["last_hidden_state"] # Shape is [batch_size, seq_len, 768]
        attn = self.attn_head(last_hidden_state)            # Shape is [batch_size, 768]
        output = self.regressor(attn)                       # Shape is [batch_size, 1]       
        return output    
    

class MyModel_ConcatLast4Layers(nn.Module):
            
    def __init__(self):        
        super().__init__() 
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.regressor = nn.Linear(768*4, 1)     
    
    def forward(self, input_ids, attention_mask):       
        raw_output = self.roberta_model(input_ids, attention_mask, 
                                        return_dict=True, output_hidden_states=True)        
        hidden_states = raw_output["hidden_states"] 
        hidden_states = torch.stack(hidden_states) # Shape is [13, batch_size, seq_len, 768]
        concat = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1) 
                                             # Shape is [batch_size, seq_len, 768*4]
        first_token = concat[:, 0, :]        # Take only 1st token, result in shape [batch_size, 768*4]
        output = self.regressor(first_token) # Shape is [batch_size, 1]    
        return output     
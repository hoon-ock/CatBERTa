import torch
from torch import Tensor
import torch.nn as nn
import copy

class MyModel(nn.Module):
            
    def __init__(self, model):        
        super().__init__() 
        self.model = model
        self.emb_dim = model.embeddings.word_embeddings.embedding_dim
        self.regressor = nn.Linear(self.emb_dim, 1)     

    def forward(self, input_ids, attention_mask):        
        raw_output = self.model(input_ids, attention_mask, return_dict=True)        
        # pooler = raw_output["last_hidden_state"][:,0,:] # Shape is [batch_size, 768]
        pooler = raw_output["pooler_output"]    # Shape is [batch_size, 768]
        output = self.regressor(pooler)         # Shape is [batch_size, 1]
        # breakpoint()
        return output 

class AllEmbRegressor(nn.Module):
            
    def __init__(self, model):        
        super().__init__() 
        self.model = model
        self.emb_dim = model.embeddings.word_embeddings.embedding_dim
        self.seq_len = model.embeddings.position_embeddings.num_embeddings
        #self.regressor = nn.Linear(self.emb_dim, 1)
        #self.regressor2 = nn.Linear(self.seq_len, 1)     
        self.regressor = nn.Sequential(
                                        nn.Dropout(0.1),
                                        nn.Linear(self.emb_dim, self.emb_dim),
                                        nn.SiLU(),
                                        nn.Linear(self.emb_dim, 1)
                                        )    

    def forward(self, input_ids, attention_mask):        
        raw_output = self.model(input_ids, attention_mask, return_dict=True)        
        last_hidden_state = raw_output["last_hidden_state"] # Shape is [batch_size, seq_len, 768]
        output = self.regressor(last_hidden_state)         # Shape is [batch_size, seq_len, 1]
        # Shape is [batch_size, seq_len, 1] -> [batch_size, 1] by applying regressor to each token
        output = output.mean(dim=1)
        return output 


class MyModel2(nn.Module):
            
    def __init__(self, model, reinit_n_layers=3):        
        super().__init__() 
        self.model = model 
        self.regressor = nn.Linear(768, 1)
        self.reinit_n_layers = reinit_n_layers
        if reinit_n_layers > 0: self._do_reinit()  

    def _debug_reinit(self, text):
        print(f"\n{text}\nPooler:\n", self.model.pooler.dense.weight.data)        
        for i, layer in enumerate(self.model.encoder.layer[-self.reinit_n_layers:]):
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    print(f"\n{i} nn.Linear:\n", module.weight.data) 
                elif isinstance(module, nn.LayerNorm):
                    print(f"\n{i} nn.LayerNorm:\n", module.weight.data)   

    def _do_reinit(self):
        # Re-init pooler.
        self.model.pooler.dense.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
        self.model.pooler.dense.bias.data.zero_()
        for param in self.model.pooler.parameters():
            param.requires_grad = True
        
        # Re-init last n layers.
        for n in range(self.reinit_n_layers):
            self.model.encoder.layer[-(n+1)].apply(self._init_weight_and_bias) 

    def _init_weight_and_bias(self, module):                        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0) 

    def forward(self, input_ids, attention_mask):        
        raw_output = self.model(input_ids, attention_mask, return_dict=True)        
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
            
    def __init__(self, model):        
        super().__init__() 
        self.model = model #RobertaModel.from_pretrained('roberta-base')       
        self.regressor = regressionHead() #nn.Linear(768, 1)     

    def forward(self, input_ids, attention_mask):        
        raw_output = self.model(input_ids, attention_mask, return_dict=True)        
        # pooler = raw_output["last_hidden_state"][:,0,:] # Shape is [batch_size, 768]
        pooler = raw_output["pooler_output"]    # Shape is [batch_size, 768]
        output = self.regressor(pooler)         # Shape is [batch_size, 1]
        return output 


class MyModel_MLP2(nn.Module):
            
    def __init__(self, model):        
        super().__init__() 
        self.model = model #RobertaModel.from_pretrained('roberta-base')       
        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.SiLU(),
            nn.Linear(768, 1)
        )     

    def forward(self, input_ids, attention_mask):        
        raw_output = self.model(input_ids, attention_mask, return_dict=True)        
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
            
    def __init__(self, model):        
        super().__init__() 
        self.model = model #RobertaModel.from_pretrained('roberta-base')
        self.attn_head = AttentionHead(768, 512)       
        self.regressor = nn.Linear(768, 1)     
    
    def forward(self, input_ids, attention_mask):       
        raw_output = self.model(input_ids, attention_mask, return_dict=True)        
        last_hidden_state = raw_output["last_hidden_state"] # Shape is [batch_size, seq_len, 768]
        attn = self.attn_head(last_hidden_state)            # Shape is [batch_size, 768]
        output = self.regressor(attn)                       # Shape is [batch_size, 1]       
        return output    
    

class MyModel_ConcatLast4Layers(nn.Module):
            
    def __init__(self, model):        
        super().__init__() 
        self.model = model #RobertaModel.from_pretrained('roberta-base')
        self.regressor = nn.Linear(768*4, 1)     
    
    def forward(self, input_ids, attention_mask):       
        raw_output = self.model(input_ids, attention_mask, 
                                        return_dict=True, output_hidden_states=True)        
        hidden_states = raw_output["hidden_states"] 
        hidden_states = torch.stack(hidden_states) # Shape is [13, batch_size, seq_len, 768]
        concat = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1) 
                                             # Shape is [batch_size, seq_len, 768*4]
        first_token = concat[:, 0, :]        # Take only 1st token, result in shape [batch_size, 768*4]
        output = self.regressor(first_token) # Shape is [batch_size, 1]    
        return output    


class MultimodalRegressor(nn.Module):
    def __init__(self, backbone_model):
        super().__init__()
        self.transformers = nn.ModuleList([copy.deepcopy(backbone_model) for _ in range(3)])
        self.emb_dim = backbone_model.embeddings.word_embeddings.embedding_dim
        self.regressors = nn.ModuleList([nn.Linear(self.emb_dim, 1) for _ in range(3)])

    def forward(self, section_input_ids, section_attention_mask):
        # section_input_ids is a torch tensor of shape [batch_size, 3, seq_len]
        raw_outputs = []
        for i in range(3):
            raw_output = self.transformers[i](section_input_ids[i], section_attention_mask[i])
            pooler = raw_output["pooler_output"]  # Shape is [batch_size, emb_dim]
            output = self.regressors[i](pooler)  # Shape is [batch_size, 1]
            raw_outputs.append(output)
        # breakpoint()
        output = torch.sum(torch.cat(raw_outputs, dim=1), dim=1, keepdim=True) # Shape is [batch_size, 1]
        # breakpoint()
        return output 
    

class MultimodalRegressor2(nn.Module):
    def __init__(self, backbone_model):
        super().__init__()
        self.transformers = nn.ModuleList([copy.deepcopy(backbone_model) for _ in range(3)])
        self.emb_dim = backbone_model.embeddings.word_embeddings.embedding_dim
        
        layers = []
        for _ in range(3):
            layers.append(nn.Linear(self.emb_dim, self.emb_dim))
            layers.append(nn.ReLU())
        self.output_block = nn.Sequential(*layers)
        self.final_regressor = nn.Linear(self.emb_dim, 1)

    def forward(self, section_input_ids, section_attention_mask):
        # section_input_ids is a torch tensor of shape [batch_size, 3, seq_len]
        first_token_embs = []
        for i in range(3):
            raw_output = self.transformers[i](section_input_ids[i], section_attention_mask[i])
            first_token_emb = raw_output["last_hidden_state"][:, 0, :]  
            first_token_embs.append(first_token_emb)
        first_token_embs = torch.stack(first_token_embs, dim=1)  # Shape is [batch_size, 3, emb_dim]
        # breakpoint()
        output = self.output_block(first_token_embs) # Shape is [batch_size, 3, emb_dim]
        output = self.final_regressor(output) # Shape is [batch_size, 3, 1]
        output = torch.sum(output, dim=1)
        # breakpoint()
        return output 
    

class MultiTransformer(nn.Module):
    def __init__(self, backbone_model):
        super().__init__()
        self.transformers = nn.ModuleList([copy.deepcopy(backbone_model) for _ in range(3)])
        self.emb_dim = backbone_model.embeddings.word_embeddings.embedding_dim
        self.regressors = nn.ModuleList([nn.Linear(self.emb_dim, 1) for _ in range(3)])
        #self.final_transformer = final_transformer # roberta encoder & pooler layer
        self.final_encoder = copy.deepcopy(backbone_model.encoder)
        self.final_pooler = copy.deepcopy(backbone_model.pooler)
        self.regressor = nn.Linear(self.emb_dim, 1)

    def forward(self, section_input_ids, section_attention_mask):
        # section_input_ids is a torch tensor of shape [batch_size, 3, seq_len]
        pooler_tensors = []
        for i in range(3):
            raw_output = self.transformers[i](section_input_ids[i], section_attention_mask[i])
            pooler = raw_output["pooler_output"]  # Shape is [batch_size, emb_dim]
            pooler_tensors.append(pooler)

        stacked_pooler = torch.stack(pooler_tensors, dim=1) # Shape is [batch_size, 3, emb_dim]
        #final_pooler = self.final_transformer(stacked_pooler) # Shape is [batch_size, emb_dim]
        last_hidden_state = self.final_encoder(stacked_pooler)['last_hidden_state'] # Shape is [batch_size, 3, emb_dim]
        final_pooler = self.final_pooler(last_hidden_state) # Shape is [batch_size, emb_dim]
        output = self.regressor(final_pooler) # Shape is [batch_size, 1]      
        return output 
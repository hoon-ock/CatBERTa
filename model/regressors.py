import torch
from torch import Tensor
import torch.nn as nn
import copy
from transformers import RobertaModel


def freeze_layers(model, freeze_layers='all'):
    for name, param in model.named_parameters():
        if freeze_layers == 'all':
            param.requires_grad = False
        else:
            if name in freeze_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True

def set_frozen_layers(model, num_layers_to_unfreeze):
    """
    Freeze all layers except the last num_layers_to_unfreeze layers and the pooler layer of the input model.

    Args:
        model (torch.nn.Module): The RoBERTa model to be modified.
        num_layers_to_unfreeze (int): Number of layers to unfreeze, including the last two layers.

    Returns:
        None (modifies the model in-place).
    """
    # Total number of layers in the encoder
    total_layers = model.config.num_hidden_layers

    # Validate num_layers_to_unfreeze to avoid errors
    if num_layers_to_unfreeze >= total_layers:
        raise ValueError("The number of layers to unfreeze must be less than the total number of layers.")

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Calculate the starting index of the layers to unfreeze
    start_index = total_layers - num_layers_to_unfreeze

    # Unfreeze the specified number of layers (last num_layers_to_unfreeze layers) in the encoder
    for param in model.encoder.layer[start_index:].parameters():
        param.requires_grad = True

    # Unfreeze the pooler layer
    for param in model.pooler.parameters():
        param.requires_grad = True

class MyModel(nn.Module):
            
    def __init__(self, model):        
        super().__init__() 
        self.model = model
        self.emb_dim = model.embeddings.word_embeddings.embedding_dim
        self.regressor = nn.Linear(self.emb_dim, 1)     

    def forward(self, input_ids, attention_mask):      
        raw_output = self.model(input_ids, attention_mask, return_dict=True)        
        # pooler = raw_output["last_hidden_state"][:,0,:] # [batch_size, 768]
        pooler = raw_output["pooler_output"]    # [batch_size, 768]
        output = self.regressor(pooler)         # [batch_size, 1]
        # breakpoint()
        return output 


class MyRegressor1(nn.Module):
    def __init__(self, model):        
        super().__init__() 
        self.model = model
        self.emb_dim = model.embeddings.word_embeddings.embedding_dim
        self.regressor = nn.Sequential( 
                                        nn.Linear(self.emb_dim, 512),
                                        # instance norm
                                        nn.LayerNorm(512),
                                        nn.GELU(),
                                        nn.Linear(512, 1)
                                        ) 
        #nn.init.xavier_normal_(self.regressor.weight)
        # reinit regressor weights with xavier normal
        


    def forward(self, input_ids, attention_mask):        
        raw_output = self.model(input_ids, attention_mask, return_dict=True)        
        output = raw_output["last_hidden_state"][:,0,:] # [batch_size, 768]
        output = self.regressor(output)         # [batch_size, 1]
        return output 
    
class MyRegressor2(nn.Module):
    def __init__(self, model):        
        super().__init__() 
        self.model = model
        self.emb_dim = model.embeddings.word_embeddings.embedding_dim
        self.comp_regressor = nn.Linear(self.emb_dim, 512) 
        self.bilinear = nn.Bilinear(self.emb_dim, 512, 1)
        self.activation = nn.ELU() 
    def forward(self, input_ids, attention_mask):        
        raw_output = self.model(input_ids, attention_mask, return_dict=True)        
        output = raw_output["last_hidden_state"][:,0,:] # [batch_size, 768]
        output = self.activation(output)         # [batch_size, 768]
        comp_output = self.comp_regressor(output)         # [batch_size, 512]
        comp_output = self.activation(comp_output)         # [batch_size, 512]
        output = self.bilinear(output, comp_output)         # [batch_size, 1]
        return output 


class MyRegressor3(nn.Module):
    def __init__(self, model, reinit_n_layers=3):        
        super().__init__() 
        self.model = model
        self.emb_dim = model.embeddings.word_embeddings.embedding_dim
        self.regressor = nn.Sequential( 
                                        nn.Linear(self.emb_dim, 512),
                                        nn.LayerNorm(512),
                                        nn.GELU(),
                                        nn.Linear(512, 1)
                                        ) 
        self.reinit_n_layers = reinit_n_layers
        if reinit_n_layers > 0: self._do_reinit()            
        
    def _do_reinit(self):
        # Re-init last n layers.
        for n in range(self.reinit_n_layers):
            self.model.encoder.layer[-(n+1)].apply(self._init_weight_and_bias)
        # Re-init regressor
        self.regressor.apply(self._init_weight_and_bias)
    
    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, input_ids, attention_mask):        
        raw_output = self.model(input_ids, attention_mask, return_dict=True)        
        output = raw_output["last_hidden_state"][:,0,:] # [batch_size, 768]
        output = self.regressor(output)         # [batch_size, 1]
        return output 

class AvgPoolRegressor(nn.Module):
            
    def __init__(self, model):        
        super().__init__() 
        self.model = model
        self.emb_dim = model.embeddings.word_embeddings.embedding_dim
        self.regressor = nn.Linear(self.emb_dim, 1)     

    def forward(self, input_ids, attention_mask):        
        raw_output = self.model(input_ids, attention_mask, return_dict=True)        
        output = raw_output["last_hidden_state"] # [batch_size, pos, emb_dim]
        output = torch.mean(output, dim=1) # [batch_size, emb_dim]
        output = self.regressor(output)         # [batch_size, 1]
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
        last_hidden_state = raw_output["last_hidden_state"] # [batch_size, seq_len, 768]
        output = self.regressor(last_hidden_state)         # [batch_size, seq_len, 1]
        # [batch_size, seq_len, 1] -> [batch_size, 1] by applying regressor to each token
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
        pooler = raw_output["pooler_output"]    # [batch_size, 768]
        output = self.regressor(pooler)         # [batch_size, 1]
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
        # pooler = raw_output["last_hidden_state"][:,0,:] # [batch_size, 768]
        pooler = raw_output["pooler_output"]    # [batch_size, 768]
        output = self.regressor(pooler)         # [batch_size, 1]
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
        pooler = raw_output["pooler_output"]    # [batch_size, 768]
        output = self.regressor(pooler)         # [batch_size, 1]
        return output 

class AttentionHead(nn.Module):
    
    def __init__(self, emb_dim=768, seq_len=512):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, seq_len)
        self.linear2 = nn.Linear(seq_len, 1)

    def forward(self, last_hidden_state):
        """
        Note:
        "last_hidden_state" [batch_size, seq_len, emb_dim].
        The "weights" produced from softmax will add up to 1 across all tokens in each record.
        """        
        linear1_output = self.linear1(last_hidden_state)  # [batch_size, seq_len, emb_dim]  
        activation = torch.tanh(linear1_output)           # [batch_size, seq_len, emb_dim]        
        score = self.linear2(activation)                  # [batch_size, seq_len, 1]        
        weights = torch.softmax(score, dim=1)             # [batch_size, seq_len, 1]              
        result = torch.sum(weights * last_hidden_state, dim=1) # [batch_size, 768]          
        return result
    

class MyModel_AttnHead(nn.Module):
            
    def __init__(self, model):        
        super().__init__() 
        self.model = model #RobertaModel.from_pretrained('roberta-base')
        self.attn_head = AttentionHead(768, 512)       
        self.regressor = nn.Linear(768, 1)     
    
    def forward(self, input_ids, attention_mask):       
        raw_output = self.model(input_ids, attention_mask, return_dict=True)        
        last_hidden_state = raw_output["last_hidden_state"] # [batch_size, seq_len, 768]
        attn = self.attn_head(last_hidden_state)            # [batch_size, 768]
        output = self.regressor(attn)                       # [batch_size, 1]       
        # breakpoint()
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
        hidden_states = torch.stack(hidden_states) # [13, batch_size, seq_len, 768]
        concat = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1) 
                                             # [batch_size, seq_len, 768*4]
        first_token = concat[:, 0, :]        # Take only 1st token, result in shape [batch_size, 768*4]
        output = self.regressor(first_token) # [batch_size, 1]    
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
            pooler = raw_output["pooler_output"]  # [batch_size, emb_dim]
            output = self.regressors[i](pooler)  # [batch_size, 1]
            raw_outputs.append(output)
        # breakpoint()
        output = torch.sum(torch.cat(raw_outputs, dim=1), dim=1, keepdim=True) # [batch_size, 1]
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
        # section_input_ids is a torch tensor of shape [3, batch_size, seq_len]
        first_token_embs = []
        for i in range(3):
            raw_output = self.transformers[i](section_input_ids[i], section_attention_mask[i])
            first_token_emb = raw_output["last_hidden_state"][:, 0, :] # [batch_size, emb_dim] 
            first_token_embs.append(first_token_emb)
        first_token_embs = torch.stack(first_token_embs, dim=1)  # [batch_size, 3, emb_dim]
        output = self.output_block(first_token_embs) # [batch_size, 3, emb_dim]
        output = self.final_regressor(output) # [batch_size, 3, 1]
        output = torch.sum(output, dim=1)   # [batch_size, 1]
        return output 


class MultimodalRegressor3(nn.Module):
    def __init__(self, backbone_model):
        super().__init__()
        #self.transformers = nn.ModuleList([copy.deepcopy(backbone_model) for _ in range(3)])
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        freeze_layers(self.roberta)
        self.backbone = backbone_model
        #self.transformers = nn.ModeleList([backbone_model, freeze_layers(self.roberta), freeze_layers(self.roberta)]) 
        self.backbone_emb_dim = backbone_model.embeddings.word_embeddings.embedding_dim
        self.roberta_emb_dim = self.roberta.embeddings.word_embeddings.embedding_dim
        self.emb_dim = self.backbone_emb_dim + self.roberta_emb_dim*2
        self.output_block = nn.Sequential(
                                        # nn.Dropout(0.1), 
                                        #   nn.Linear(self.emb_dim, self.emb_dim),
                                        #   nn.SiLU(),
                                          nn.Linear(self.emb_dim, 1)
                                        )

    def forward(self, section_input_ids, section_attention_mask):
        sys_emb = self.backbone(section_input_ids[0], section_attention_mask[0])['pooler_output'] #["last_hidden_state"][:, 0, :] # [batch_size, emb_dim]
        ads_emb = self.roberta(section_input_ids[1], section_attention_mask[1])['pooler_output'] #["last_hidden_state"][:, 0, :] # [batch_size, emb_dim]
        bulk_emb = self.roberta(section_input_ids[2], section_attention_mask[2])['pooler_output'] #["last_hidden_state"][:, 0, :] # [batch_size, emb_dim]
        #breakpoint()
        emb = torch.cat([sys_emb, ads_emb, bulk_emb], dim=-1) # [batch_size, 3*emb_dim]
        output = self.output_block(emb) # [batch_size, 1]
        #breakpoint()
        return output 

class MultimodalRegressor4(nn.Module):
    def __init__(self, backbone_model):
        super().__init__()
        self.roberta1 = RobertaModel.from_pretrained('roberta-base')
        self.roberta2 = RobertaModel.from_pretrained('roberta-base')
        #set_frozen_layers(self.roberta, 2)
        self.backbone = backbone_model
        set_frozen_layers(self.backbone, 2)
        self.backbone_emb_dim = backbone_model.embeddings.word_embeddings.embedding_dim
        self.roberta_emb_dim = self.roberta1.embeddings.word_embeddings.embedding_dim
        self.emb_dim = self.backbone_emb_dim + self.roberta_emb_dim*2
        self.output_block = nn.Linear(self.emb_dim, 1)

    def forward(self, section_input_ids, section_attention_mask):
        sys_emb = self.backbone(section_input_ids[0], section_attention_mask[0])['pooler_output'] # [batch_size, emb_dim]
        ads_emb = self.roberta1(section_input_ids[1], section_attention_mask[1])['pooler_output'] # [batch_size, emb_dim]
        bulk_emb = self.roberta2(section_input_ids[2], section_attention_mask[2])['pooler_output'] # [batch_size, emb_dim]
        emb = torch.cat([sys_emb, ads_emb, bulk_emb], dim=-1) # [batch_size, 3*emb_dim]
        output = self.output_block(emb) # [batch_size, 1]
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
            pooler = raw_output["pooler_output"]  # [batch_size, emb_dim]
            pooler_tensors.append(pooler)

        stacked_pooler = torch.stack(pooler_tensors, dim=1) # [batch_size, 3, emb_dim]
        #final_pooler = self.final_transformer(stacked_pooler) # [batch_size, emb_dim]
        last_hidden_state = self.final_encoder(stacked_pooler)['last_hidden_state'] # [batch_size, 3, emb_dim]
        final_pooler = self.final_pooler(last_hidden_state) # [batch_size, emb_dim]
        output = self.regressor(final_pooler) # [batch_size, 1]      
        return output 


class YModel(nn.Module):
    def __init__(self, pretrained_model, sub_backbone, reinit_n_layers=3):
        super().__init__()
        self.pretrained_backbone = pretrained_model.model  # Frozen pretrained backbone model
        self.pretrained_regressor = pretrained_model.regressor  # Frozen pretrained regressor model
        self._freeze_pretrained()

        self.sub_backbone = sub_backbone
        self.emb_dim = self.pretrained_backbone.embeddings.word_embeddings.embedding_dim
        assert self.emb_dim == sub_backbone.embeddings.word_embeddings.embedding_dim, \
            "Embedding dimensions of the two models must be the same!"
        # Build similarity matrix
        # self.similarity = nn.CosineSimilarity(dim=1)
        self.sim_loss = nn.CosineEmbeddingLoss()
        self.regressor = nn.Linear(self.emb_dim, 1)
        self.reinit_n_layers = reinit_n_layers
        if reinit_n_layers > 0: self._do_reinit(), print("Reinit last {} layers.".format(reinit_n_layers))            
           
    def _do_reinit(self):
        # Re-init last n layers.
        for n in range(self.reinit_n_layers):
            self.sub_backbone.encoder.layer[-(n+1)].apply(self._init_weight_and_bias)
        # Re-init regressor
        self.regressor.apply(self._init_weight_and_bias)
    
    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _freeze_pretrained(self):
        for param in self.pretrained_backbone.parameters():
            param.requires_grad = False
        # for param in self.pretrained_regressor.parameters():
        #     param.requires_grad = False
       
    def forward(self, input_ids, attention_mask):
        # Compute embeddings and pooled representations from pretrained backbone model
        # with torch.no_grad():
        pretrained_output = self.pretrained_backbone(input_ids, attention_mask, return_dict=True)
        pretrained_embedding = pretrained_output.last_hidden_state[:, 0, :]  # [batch_size, emb_dim]
        pretrained_pooler = pretrained_output.pooler_output  # [batch_size, emb_dim]
        pretrained_regression = self.pretrained_regressor(pretrained_pooler)  # [batch_size, 1]
        
        # Compute embeddings and pooled representations from sub-backbone model
        sub_output = self.sub_backbone(input_ids, attention_mask, return_dict=True)
        sub_embedding = sub_output.last_hidden_state[:, 0, :]  # [batch_size, emb_dim]
        sub_pooler = sub_output.pooler_output  # [batch_size, emb_dim]
        
        # Calculate cosine similarity between embeddings
        # sim = self.similarity(pretrained_embedding, sub_embedding).unsqueeze(1)  # [batch_size, 1]
        loss_sim = self.sim_loss(pretrained_embedding, sub_embedding, -torch.ones(input_ids.shape[0]).to(input_ids.device))
        
        # Regression using pooled representations
        # pretrained_regression = self.pretrained_regressor(pretrained_pooler)
        sub_regression = self.regressor(sub_pooler)
        output = pretrained_regression + sub_regression  # [batch_size, 1]
        # breakpoint()
        # loss_sim = 0
        # output = pretrained_regression
        return output, loss_sim
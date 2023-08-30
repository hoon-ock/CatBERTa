import torch.nn as nn


def freeze_layers(model, freeze_layers='all'):
    """
    Freeze or unfreeze layers in a neural network model.

    Args:
        model (nn.Module): The neural network model whose layers are to be frozen or unfrozen.
        freeze_layers (str or list): Determines which layers to freeze.
            - If 'all', all layers will be frozen.
            - If a list of layer names, only the specified layers will be frozen.
    """
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

class PoolerRegressor(nn.Module):
    '''
    A regressor built on top of a transformer model's pooler output.

    This class encapsulates a regressor that takes the pooler output of a transformer model
    and predicts a scalar value. It's designed for regression tasks.

    Args:
        model (nn.Module): The transformer model to use for feature extraction.

    Attributes:
        model (nn.Module): The underlying transformer model.
        emb_dim (int): The dimension of the embedding used by the transformer model.
        regressor (nn.Linear): Linear layer for regression prediction.
    '''
            
    def __init__(self, model):        
        super().__init__() 
        self.model = model
        self.emb_dim = model.embeddings.word_embeddings.embedding_dim
        self.regressor = nn.Linear(self.emb_dim, 1)     

    def forward(self, input_ids, attention_mask):      
        raw_output = self.model(input_ids, attention_mask, return_dict=True)        
        pooler = raw_output["pooler_output"]    # [batch_size, 768]
        output = self.regressor(pooler)         # [batch_size, 1]
        return output 


class MyRegressor1(nn.Module):
    def __init__(self, model):        
        super().__init__() 
        self.model = model
        self.emb_dim = model.embeddings.word_embeddings.embedding_dim
        self.regressor = nn.Sequential( 
                                        nn.Linear(self.emb_dim, 512),
                                        nn.LayerNorm(512),
                                        nn.GELU(),
                                        nn.Linear(512, 1)
                                        ) 
        nn.init.xavier_normal_(self.regressor.weight)  # reinit regressor weights with xavier normal
        
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
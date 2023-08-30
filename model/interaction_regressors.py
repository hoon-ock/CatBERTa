import torch
import torch.nn as nn
import math
import copy 

class InteractionBlock(nn.Module):
    def __init__(self, emb_dim):
        super(InteractionBlock, self).__init__()
        self.W_q = nn.Linear(emb_dim, emb_dim)
        self.W_k = nn.Linear(emb_dim, emb_dim)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, query, key, value):
        assert query.size() == key.size() == value.size(), "query, key, and value must have the same size"
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)
        
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, value)
        return output

class SectionInteractionBlock(nn.Module):
    def __init__(self, emb_dim):
        super(SectionInteractionBlock, self).__init__()
        self.interaction_block1 = InteractionBlock(emb_dim)
        self.interaction_block2 = InteractionBlock(emb_dim)
    
    def forward(self, inputs):
        # inputs: (batch_size, num_sections, hidden_size)
        system = inputs[:, 0, :].unsqueeze(1)
        adsorbate = inputs[:, 1, :].unsqueeze(1)
        bulk = inputs[:, 2, :].unsqueeze(1)

        system_adsorbate = self.interaction_block1(system, adsorbate, adsorbate)
        system_bulk = self.interaction_block2(system, bulk, bulk)
        output = torch.cat([system, system_adsorbate, system_bulk], dim=1) # (batch_size, num_sections, hidden_size)
        return output

class OutputBlock(nn.Module):
    def __init__(self, emb_dim):
        super(OutputBlock, self).__init__()
        self.emb_dim = emb_dim
        
        layers = []
        for _ in range(3):
            layers.append(nn.Linear(self.emb_dim, self.emb_dim))
            layers.append(nn.ReLU())
        self.output_block = nn.Sequential(*layers)
        self.final_regressor = nn.Linear(self.emb_dim, 1)
        
    def forward(self, x):
        x = self.output_block(x)
        output = self.final_regressor(x)
        return output


class InteractionRegressor(nn.Module):
    def __init__(self, backbone_model):
        super(InteractionRegressor, self).__init__()
        self.transformers = nn.ModuleList([copy.deepcopy(backbone_model) for _ in range(3)])
        self.emb_dim = backbone_model.embeddings.word_embeddings.embedding_dim
        self.interaction_block = SectionInteractionBlock(self.emb_dim)
        self.output_block = OutputBlock(self.emb_dim)
        # self.final_regressor = nn.Linear(3, 1)

    def forward(self, section_input_ids, section_attention_mask):
        # section_input_ids is a torch tensor of shape [batch_size, 3, seq_len]
        first_token_embs = []
        for i in range(3):
            raw_output = self.transformers[i](section_input_ids[i], section_attention_mask[i])
            first_token_emb = raw_output["last_hidden_state"][:, 0, :]  # [batch_size, emb_dim]
            first_token_embs.append(first_token_emb)
        first_token_embs = torch.stack(first_token_embs, dim=1)  # [batch_size, 3, emb_dim]
        output = self.interaction_block(first_token_embs) # [batch_size, 3, emb_dim]
        output = self.output_block(output) # [batch_size, 3, 1]
        # breakpoint()
        output = torch.sum(output, dim=1) # [batch_size, 1]
        # output = self.final_regressor(output) # [batch_size, 1]
        # breakpoint()
        return output
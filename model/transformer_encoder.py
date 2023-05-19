import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math 

class Transformer(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, num_heads, dropout):
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_encoding = PositionalEncoding(emb_size, dropout)
        self.transformer = nn.Transformer(d_model=emb_size, 
                                          nhead=num_heads, 
                                          num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers, 
                                          dim_feedforward=hidden_size, 
                                          dropout=dropout)
        self.fc = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len)
        #print(x.shape)
        #import pdb; pdb.set_trace()
        x = self.embedding(x) # (batch_size, seq_len, emb_size)
        x = self.pos_encoding(x) # (batch_size, seq_len, emb_size)
        # Transformer expects the input in the shape of (seq_len, batch_size, emb_size)
        x = x.transpose(0, 1)
        x = self.transformer(x, x) # source, target (seq_len, batch_size, emb_size)
        # Flatten the output for the fully connected layer
        x = x.transpose(0, 1).contiguous()
        x = x.view(-1, x.size(-1))
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
import math

import torch
from torch import nn


class SingleHeadAttentionLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(query_size, value_size)

    def forward(self, q, k, v):
        query = self.dense_q(q)
        key = self.dense_k(k)
        value = self.dense_v(v)
        g = torch.div(torch.matmul(query, key.T), math.sqrt(self.attention_size))
        score = torch.softmax(g, dim=-1)
        output = torch.sum(torch.unsqueeze(score, dim=-1) * value, dim=-2)
        return output


def masked_softmax(X, mask):  #@save
    if not(mask is None):
        mask = mask.bool()
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, mask, value=0):
        # maxlen = X.size(1)
        # mask = torch.arange((maxlen), dtype=torch.float32,
        #                     device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if mask is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        # if valid_lens.dim() == 1:
        #     valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        # else:
        #     valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        # print(X.dtype, mask.dtype)
        X = _sequence_mask(X.reshape(-1, shape[-1]), mask, value=-1e+4)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    
class CustomAttentionLayer(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self,query_size, key_size, value_size, attention_size, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(attention_size,value_size)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    def forward(self, queries, keys, values,mask=None):
        queries = queries.unsqueeze(1)
        # print('first queries: ',queries)
        queries = self.dense_q(queries)
        # print('2nd queries: ',queries)

        keys = self.dense_k(keys)
        # print('keys: ',keys)
        # values = self.dense_v(values)
        # print(queries.shape, keys.shape)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # print('features', features)
        features = torch.tanh(features)
        # print('2nd features', features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)

        scores = self.dense_v(features).squeeze(-1)
        # print('score: ',scores)
        self.attention_weights = masked_softmax(scores, mask)
        # print('attn_weights: ', self.attention_weights)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        att = torch.bmm(self.dropout(self.attention_weights), values) 

        # print('att: ',att)
        att = att.squeeze(1)
        # print('2nd att: ' , att.shape)
        return att   
    
class DotProductAttention(nn.Module):
    def __init__(self, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.context = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(attention_size, 1)))
        self.dense = nn.Linear(value_size, attention_size)

    def forward(self, x):
        t = self.dense(x)
        vu = torch.matmul(t, self.context).squeeze()
        score = torch.softmax(vu, dim=-1)
        output = torch.sum(x * torch.unsqueeze(score, dim=-1), dim=-2)
        return output


import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, query_dim, key_dim, value_dim, dropout = 0.1):
        super().__init__()
        
        self.scale = 1.0/(np.sqrt(query_dim ** 2) )
        self.softmax = nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)

        attention_dim = key_dim
        self.dense_q = nn.Linear(query_dim, attention_dim)
        self.dense_k = nn.Linear(key_dim, attention_dim)
        self.dense_v = nn.Linear(attention_dim,value_dim)


    def forward(self, queries, keys, values, mask = None, sequence_key = False):
        # query: [B,Q] (hidden state, decoder output, etc.)
        # keys: [T,B,K] (encoder outputs)
        # values: [T,B,V] (encoder outputs)
        # assume Q == K
        # print('mask', mask.shape)
        # compute energy
        if not sequence_key:
            query = queries.unsqueeze(1) # [B,Q] -> [B,1,Q]
        else:
            query = queries
        # print(query.shape)
        # keys = keys.permute(1,2,0) # [T,B,K] -> [B,K,T]
        query = self.dense_q(query)
        keys = self.dense_k(keys)
        keys = keys.permute(0,2,1)
        # print(query.shape, keys.shape)

        # print(query.shape, keys.shape, values.shape)

        energy = torch.bmm(query, keys) # [B,1,Q]*[B,K,T] = [B,1,T]
        energy = self.softmax(energy.mul_(self.scale))
        # print(query.shape, energy.shape, mask.shape)
        if mask != None:
            # apply mask, renormalize
            if not sequence_key:
                energy = energy.squeeze(1) *mask
                energy = energy.unsqueeze(1)

                energy.div(energy.sum(2, keepdim=True))
            else:
                energy = energy * mask.unsqueeze(1).repeat(1,energy.shape[1],1)
                # print(energy.shape)

        # weight values
        # values = values.transpose(0,1) # [T,B,V] -> [B,T,V]
        # print('combo',torch.bmm(energy, values).shape)
        combo =  torch.bmm(energy, values)
        if not sequence_key:
            combo = combo.squeeze(1) # [B,1,T]*[B,T,V] -> [B,V]
        # else:
        #     combo = 

        # print(combo.shape)

        return combo
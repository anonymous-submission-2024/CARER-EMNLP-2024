import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):

        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)
        self.max_pos = max_seq_len

    def forward(self, input_len):

        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        pos = np.zeros([input_len.size(0), self.max_pos])
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                pos[ind, pos_ind - 1] = pos_ind
        input_pos = tensor(pos)
        return self.position_encoding(input_pos), input_pos


class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths=None):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if mask_or_lengths is not None:
            if len(mask_or_lengths.size()) == 1:
                mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(
                    1))
            else:
                mask = mask_or_lengths
            inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = inputs.max(1)[0]
        return max_pooled


class Attention(nn.Module):
    def __init__(self, in_feature, num_head, dropout):
        super(Attention, self).__init__()
        self.in_feature = in_feature
        self.num_head = num_head
        self.size_per_head = in_feature // num_head
        self.out_dim = num_head * self.size_per_head
        assert self.size_per_head * num_head == in_feature
        self.q_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.k_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.v_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.fc = nn.Linear(in_feature, in_feature, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_feature)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = key.size(0)
        res = query
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        query = query.view(batch_size, self.num_head, -1, self.size_per_head)
        key = key.view(batch_size, self.num_head, -1, self.size_per_head)
        value = value.view(batch_size, self.num_head, -1, self.size_per_head)

        scale = np.sqrt(self.size_per_head)
        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / scale
        if attn_mask is not None:
            batch_size, q_len, k_len = attn_mask.size()
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, self.num_head, q_len, k_len)
            energy = energy.masked_fill(attn_mask == 0, -np.inf)

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, value)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.in_feature)
        attention = attention.sum(dim=1).squeeze().permute(0, 2, 1) / self.num_head
        x = self.fc(x)
        x = self.dropout(x)
        x += res
        x = self.layer_norm(x)
        return x, attention


class HitaNet(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos):
        super(HitaNet, self).__init__()
        #self.embbedding = nn.Embedding(vocab_size + 1, d_model)#, padding_idx=-1)
        # self.embbedding = nn.Sequential(nn.Linear(vocab_size + 1, d_model), nn.ReLU())
        self.embbedding1 =  nn.Sequential(nn.Linear(vocab_size, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(d_model))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)
        self.encoder_layers = nn.ModuleList([Attention(d_model, num_heads, dropout) for _ in range(1)])
        self.positional_feed_forward_layers = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(),
                                                                           nn.Linear(4 * d_model, d_model))
                                                             for _ in range(1)])
        self.pos_emb = PositionalEncoding(d_model, max_pos)
        self.time_layer = torch.nn.Linear(64, 256)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.selection_time_layer = nn.Linear(1, 64)
        self.weight_layer = torch.nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.self_layer = torch.nn.Linear(d_model, 1)
        self.quiry_layer = torch.nn.Linear(d_model, 64)
        self.quiry_weight_layer = torch.nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
          
          
          
  
           
           
           

    def forward(self, input_seqs, masks, lengths, seq_time_step):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        # time_feature_cache = time_feature
        time_feature = self.time_layer(time_feature)
        
        x = self.embbedding1(input_seqs)
        x = self.emb_dropout(x)
        bs, seq_length, d_model = x.size()

        output_pos, ind_pos = self.pos_emb(lengths)
        x += output_pos
        x += time_feature
        attentions = []
        outputs = []
        for i in range(len(self.encoder_layers)):
            x, attention = self.encoder_layers[i](x, x, x, masks)
            res = x
            x = self.positional_feed_forward_layers[i](x)
            x = self.dropout(x)
            x = self.layer_norm(x + res)
            attentions.append(attention)
            outputs.append(x)
        final_statues = outputs[-1].gather(1, lengths[:, None, None].expand(bs, 1, d_model) - 1).expand(bs, seq_length,
                                                                                                        d_model)
        quiryes = self.relu(self.quiry_layer(final_statues))
        #quiryes = self.relu(final_statues)
        mask = (torch.arange(seq_length, device=x.device).unsqueeze(0).expand(bs, seq_length) >= lengths.unsqueeze(1))
        self_weight = torch.softmax(self.self_layer(outputs[-1]).squeeze().masked_fill(mask, -np.inf), dim=1).view(bs,
                                                                                                                    seq_length).unsqueeze(
            2)
        selection_feature = self.relu(self.weight_layer(self.selection_time_layer(seq_time_step)))
        selection_feature = torch.sum(selection_feature * quiryes, 2) / 8
        time_weight = torch.softmax(selection_feature.masked_fill(mask, -np.inf), dim=1).view(bs, seq_length).unsqueeze(
            2)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2).view(bs, seq_length, 2)
        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = outputs[-1] * total_weight.unsqueeze(2)
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        prediction = self.output_mlp(averaged_features)
        return prediction


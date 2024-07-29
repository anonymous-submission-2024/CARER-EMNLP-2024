import torch
from torch import nn

from models.utils import SingleHeadAttentionLayer
import numpy as np
from torch_geometric.nn import Sequential, GCNConv, GINConv, GATConv

import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, code_num, code_size, graph_size):
        super().__init__()
        self.code_num = code_num
        self.c_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        self.n_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        self.u_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, graph_size)))

        # if code_num == 6743:
        # # try:
        #     bert_embeddings = torch.from_numpy(np.load('data/mimic4/bert_umap_embeddings.npy'))
        # else:
        #     bert_embeddings = torch.from_numpy(np.load('data/mimic3/bert_umap_embeddings.npy'))

        # self.c_embeddings.data = bert_embeddings
        # self.n_embeddings.data = bert_embeddings
        # self.u_embeddings.data = bert_embeddings

    def forward(self):
        return self.c_embeddings, self.n_embeddings, self.u_embeddings


class GraphLayer(nn.Module):
    def __init__(self, adj, code_size, graph_size):
        super().__init__()
        self.adj = adj
        self.dense = nn.Linear(code_size, graph_size)
        self.activation = nn.LeakyReLU()

    def forward(self, code_x, neighbor, c_embeddings, n_embeddings,prior):
        center_codes = torch.unsqueeze(code_x, dim=-1)
        # print(center_codes)
        neighbor_codes = torch.unsqueeze(neighbor, dim=-1)
        center_embeddings = center_codes * c_embeddings
        neighbor_embeddings = neighbor_codes * n_embeddings

        cc_embeddings = center_codes * torch.matmul(self.adj, center_embeddings)
        cc_embeddings = cc_embeddings * torch.matmul(self.adj, cc_embeddings)

        # print(center_embeddings.shape, cc_embeddings.shape)
        cc_embeddings = self.activation(self.dense(cc_embeddings))

        return cc_embeddings

import math
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print(input.shape, self.weight.shape)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, code_x, c_embeddings, adj):

        center_codes = torch.unsqueeze(code_x, dim=-1)
        # print(center_codes)
        x = center_codes * c_embeddings

        x = self.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class ChebConvNet(nn.Module):
    def __init__(self,adj,input_dim,output_dim,hidden_dim = 4,K = 1,num_conv_layers = 1,dropout = 0.2, device = 'cuda'):
        super(ChebConvNet, self).__init__()
        self.adj  =adj
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim =hidden_dim
        self.K = K
        self.num_layers = num_conv_layers
        self.device = device
        self.convs = nn.ModuleList()
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

        assert self.num_layers >= 1, "Number of layers have to be >= 1"

        if self.num_layers == 1:
            self.convs.append(GCNConv(self.input_dim,self.output_dim).to(self.device))
        elif self.num_layers >= 2:
            self.convs.append(GCNConv(self.input_dim,self.hidden_dim).to(self.device))
            for i in range(self.num_layers - 2):
                self.convs.append(GCNConv(self.hidden_dim,self.hidden_dim).to(self.device))
            self.convs.append(GCNConv(self.hidden_dim,self.output_dim).to(self.device))
        self.dense = nn.Linear(self.output_dim, self.output_dim)
        # if self.num_layers == 1:
        #     self.convs.append(GIN(self.input_dim,self.output_dim,num_layers=1).to(self.device))
        # elif self.num_layers >= 2:
        #     self.convs.append(GIN(self.input_dim,self.hidden_dim,num_layers=1).to(self.device))
        #     for i in range(self.num_layers - 2):
        #         self.convs.append(GIN(self.hidden_dim,self.hidden_dim,num_layers=1).to(self.device))
        #     self.convs.append(GIN(self.hidden_dim,self.output_dim,num_layers=1).to(self.device))

    def forward(self, code_x,c_embeddings):
        # adj = 
        # laplacian = self.laplacian.unsqueeze(0).repeat(x.shape[0],1,1)

        edge_index = self.adj.nonzero().t().contiguous()
        row, col = edge_index
        edge_weight = self.adj[row, col]

        center_codes = torch.unsqueeze(code_x, dim=-1)
        x = center_codes * c_embeddings
        # print(x.shape)

        # for i in range(self.num_layers - 1):
        #     x = self.dropout(x)
        #     x = self.convs[i](x,edge_index,edge_weight)
        #     x = self.relu(x)

        # x = self.dropout(x)
        # print('x_before',x)
        # print(x.shape)
        # print(self.input_dim,self.output_dim)
        # x = self.relu(x)
        x = self.convs[-1](x,edge_index,edge_weight)
        x = self.relu(x)
        x = center_codes * x
        cc_embeddings = self.relu(self.dense(x))

        # cc_embeddings = self.relu(self.dense(cc_embeddings))

        # x = global_add_pool(x, batch)
        # print('x_after',x)
        # x = self.softmax(x)
        # print('x_after_softmax',x)
        return cc_embeddings

class LSTM(nn.Module):

    def __init__(self, input_size,  hidden_size = 256, output_size = 1, num_layers  =1, bidirect = True, device = 'cuda'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # print(self.hidden_size)
        self.num_layers = num_layers
        self.bidirect = bidirect
        self.output_size = output_size
        
        self.D = 2 if self.bidirect else 1
        self.device = device
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True
        ).to(self.device)
        self.fc = nn.Linear(in_features=self.hidden_size * 2, out_features=self.output_size).to(self.device)
        self.relu = nn.LeakyReLU(0.1)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        h_0 = torch.zeros(self.D * self.num_layers,  self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.D * self.num_layers, self.hidden_size).to(self.device)
        # Propagate input through LSTM
        # print(x.de)
        # print(x.shape,h_0.shape,c_0.shape)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn[0].view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        # print(output.shape)
        # out = self.relu(hn)
        # print(out.shape)
        # out = self.relu(output.mean(dim = 1))
        # print(out)
        # print(out)
        # print(out.shape)
        # print(output.shape)
        # out = self.fc(out)  # Final Output
        # # print(out)
        # out  = self.sigmoid(out)
        # print(self.fc.weight.grad)
        # print(out)
        # print(out.shape)
        return output


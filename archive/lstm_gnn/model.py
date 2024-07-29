import torch.nn as nn
import torch
from models.lstm_gnn.layers import ChebConvNet, ChebConvLayer, LSTM, DotProductAttention
def get_laplacian_matrix(adj, normalize_L = True):
    r"""
    Get the Laplacian/normalized Laplacian matrices from a batch of adjacency matrices
    Parameters
    --------------
        adj: tensor(..., N, N)
            Batches of symmetric adjacency matrices
        
        normalize_L: bool
            Whether to normalize the Laplacian matrix
            If `False`, then `L = D - A`
            If `True`, then `L = D^-1 (D - A)`
    Returns
    -------------
        L: tensor(..., N, N)
            Resulting Laplacian matrix
    """

    # Apply the equation L = D - A
    N = adj.shape[-1]
    arr = torch.arange(N)
    L = -adj
    D = torch.sum(adj, dim=-1)
    L[..., arr, arr] = D

    # Normalize by the degree : L = D^-1 (D - A)
    if normalize_L:
        Dinv = torch.zeros_like(L)
        Dinv[..., arr, arr] = D ** -1
        L = torch.matmul(Dinv, L)

    return L


class GC_LSTM(nn.Module):
    """ The main model for prediction.
    Input: graph signal X: (batch_size, input_time_steps,code_num, num_features)
              Laplacian: (batch_size,input_time_steps,code_num,code_num)
       Output: The predicted PM 2.5 index: (batch_size, output_time_steps)"""
    def __init__(self,code_adj,code_num,output_dim  = 1,
                 code_embedding_dim = 64, graph_input_dim  = 64,
                 graph_hidden_dim = 64, graph_output_dim = 64,
                    hidden_lstm_dim = 64, 
                 num_conv_layers = 1,input_len = 100,
                 dropout = 0.2, device = 'cuda'):
        super(GC_LSTM, self).__init__()
        self.code_adj = code_adj
        # self.laplacian = get_laplacian_matrix(code_adj)
        self.code_num = code_num
        self.code_embedding_dim = code_embedding_dim
        self.output_dim = output_dim
        self.graph_input_dim = graph_input_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.graph_output_dim = graph_output_dim
        self.hidden_lstm_dim = hidden_lstm_dim
        self.K = 1
        self.num_conv_layers = 1
        self.num_input_steps = input_len
        self.dropout = dropout
        self.device = device
        # self.gcns = nn.ModuleList()

        # Declare a graph convolution layer for each input time step t
        # for i in range(self.num_input_steps):
        #     self.gcns.append(ChebConvNet( device=self.device))
        self.gcn = ChebConvNet(adj=self.code_adj,input_dim  = self.graph_input_dim, 
                               output_dim = self.graph_output_dim,
                               hidden_dim = self.graph_hidden_dim, 
                               K=self.K,
                                num_conv_layers = self.num_conv_layers, 
                                 dropout = self.dropout, device=self.device)
        self.lstm = LSTM(input_size =self.graph_output_dim + self.code_embedding_dim, hidden_size = self.hidden_lstm_dim, 
                         output_size = self.output_dim, num_layers = 1,
                           device=self.device)

        self.embedding = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(self.code_num, 
                                                                               self.code_embedding_dim))).to(self.device)
        # self.embedding_dense = nn.Linear(self.graph_output_dim, self.graph_output_dim).to(self.device)
        import numpy as np
        bert_embeddings = torch.from_numpy(np.load('data/mimic3/bert_umap_embeddings.npy'))
        self.embeddings = bert_embeddings

        self.embedding_dense = nn.Linear(self.code_embedding_dim, self.graph_output_dim).to(self.device)

        self.relu = nn.LeakyReLU(0.5)

        self.attention = DotProductAttention(value_size=self.hidden_lstm_dim * 2,attention_size=256,device = self.device)
        self.ht_attention = DotProductAttention(value_size=graph_output_dim, attention_size=256, device=self.device)
        self.xt_attention = DotProductAttention(value_size=code_embedding_dim, attention_size=64, device=self.device)

        self.output_fc = nn.Linear(self.hidden_lstm_dim * 2,1).to(self.device)
        self.sigmoid = nn.Sigmoid()

    # x: (batch_size, input_time_steps, num_station, num_features)
    def forward(self, x,  visit_lens):
        num_input_steps = x.shape[1]
        lstm_inputs = []
        # print(self.embedding.data)

        for t in range(num_input_steps):
            # batch_wise operation xt: (batch_size,num_time_steps, N, num_features)
            # print(x)
            xt = x[:, t, :]
            # print(xt.shape)
            # print(xt.nonzero().shape)
            non_zero_vector = xt.nonzero()
            if non_zero_vector.shape[0] != 0:
                # print(xt.nonzero())
                batch_non_zero_dict = {}
                for non_zero_rows in non_zero_vector:
                    # print(non_zero_rows[0].item(), non_zero_rows[1].item())
                    batch_id, batch_non_zero = non_zero_rows[0].item(), non_zero_rows[1].item()
                    if batch_id not in batch_non_zero_dict:
                        batch_non_zero_dict[batch_id] = [batch_non_zero]
                    else:
                        batch_non_zero_dict[batch_id].append(batch_non_zero)

                    # n
                # print(batch_non_zero_dict)
                xt = torch.unsqueeze(xt,dim=-1)
                # print(xt)

                # print(xt.shape, self.embedding.shape)
                xt = xt * self.embedding
                ht = self.gcn(xt)
            
                
                average_ht = []
                average_xt = []
                for batch_id in range(ht.shape[0]):
                    try:
                        non_zero_batch = batch_non_zero_dict[batch_id]
                        non_zero_embedings = ht[batch_id][non_zero_batch]
                        average_batch_embeddings=  non_zero_embedings.mean(dim = 0)
                        average_ht.append(average_batch_embeddings)
        
                        non_zero_embedings = xt[batch_id][non_zero_batch]
                        average_batch_embeddings=  non_zero_embedings.mean(dim = 0)
                        average_xt.append(average_batch_embeddings)

                    except:
                        # print(1)
                        average_ht.append(torch.zeros(self.graph_output_dim).to(self.device))
                        average_xt.append(torch.zeros(self.code_embedding_dim).to(self.device))
                # ht = self.ht_attention(ht)
                # xt = self.xt_attention(xt)

                ht = torch.stack(average_ht,dim = 0)
                xt = torch.stack(average_xt,dim = 0)
                # print(xt)

                # ht = self.relu(self.embedding_dense(ht))

                # ht = average_ht.mean(dim = 0)
                # print(average_ht.shape)
                    # print(non_zero_embedings,non_zero_embedings.shape)
                # ht = ht.sum(dim = 1)
                # print(ht)

                # print(ht)
                # print(ht.shape)
                # Concatenate the graph signal Xt and the graph convolution output Ht to be LSTM inputs
                # lstm_input = torch.cat((xt, ht), dim=2)
                lstm_input = torch.cat([ht,xt], dim =-1)
                # print(lstm_input)
                # print(lstm_input.shape)
                # Flatten the concatenated vector
                # lstm_input = lstm_input.view(lstm_input.shape[0], lstm_input.shape[1] * lstm_input.shape[2])
                # Stack all lstm inputs at different time steps
                lstm_inputs.append(lstm_input)
            else:
                break
        lstm_inputs = torch.stack(lstm_inputs, dim=1)
        # print(lstm_inputs.shape)
        
        # print(lstm_inputs.shape)
        # Pass through lstm  + fc layer
        lstm_output = self.lstm(lstm_inputs)
        # print(self.hidden_lstm_dim)
        # print(lstm_output)
        # lstm_output = lstm_inputs
        output = self.attention(lstm_output)
        # print(output)
        output = self.relu(output)
        # print(output)
        # print(output.shape)
        output  = self.sigmoid(self.output_fc(output))
        # print(output.shape)

        return output

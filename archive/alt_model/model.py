import torch
from torch import nn

from models.alt_model.layers import EmbeddingLayer, GraphLayer,  LSTM, ChebConvNet, GraphConvolution, GCN
from models.alt_model.utils import DotProductAttention


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0., activation=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


class Model(nn.Module):
    def __init__(self, code_num, code_size,
                 adj, graph_size, hidden_size, t_attention_size, t_output_size,
                 output_size, dropout_rate, activation):
        super().__init__()
        self.adj = adj
        code_size = 64
        graph_size = 128
        hidden_size = 128
        t_attention_size = 256
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
        self.graph_layer = GCN(nfeat= code_size, nhid = graph_size,dropout = 0.3).to('cuda')

        # self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, t_attention_size, t_output_size)
        self.lstm = LSTM(input_size=graph_size * 30,hidden_size=hidden_size)
        self.attention = DotProductAttention(hidden_size, t_attention_size)

        self.code_attention = DotProductAttention(graph_size,graph_size)

        # self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)
        self.classifier = Classifier(128, output_size, dropout_rate, activation)
        # self.graph_layer = ChebConvNet(adj=adj,
        #                        input_dim  = code_size,
        #                        output_dim = graph_size,
        #                        hidden_dim = graph_size, 
        #                        K=1,
        #                         num_conv_layers = 1, 
        #                          dropout = dropout_rate, device='cuda')


        self.prior_fc = nn.Linear(1,graph_size)
        self.relu = nn.LeakyReLU()
        self.layernorm = nn.LayerNorm(graph_size)


        self.lstm_cell = nn.LSTMCell(input_size = graph_size,hidden_size=hidden_size, device = 'cuda')
        self.hidden_size = hidden_size

    def forward(self, code_x, divided, neighbors, lens, prior, return_attention = False):
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings
        output = []
        # print('code_x',code_x.shape)
        i = 0
        # prior = prior.unsqueeze(1).repeat(1,64)
        prior /= prior.sum()
        prior = prior.unsqueeze(1)
    
        # print(prior)
        attention_weights = []

        for code_x_i, divided_i, neighbor_i, len_i in zip(code_x, divided, neighbors, lens):
            i +=1
            no_embeddings_i_prev = None
            output_i = []
            h_t = None
            # print(prior.shape)
            time_graph_embeddings=  []
            prev_h = torch.zeros(self.hidden_size).to('cuda')
            prev_c = torch.zeros(self.hidden_size).to('cuda')
            lstm_outputs = []
            batch_attention_weights = []
            for t, (c_it, d_it, n_it, len_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i))):
                c_it_indices = torch.nonzero(c_it).flatten()
                mask = torch.zeros(c_it.shape).to(c_it.device)
                mask[c_it_indices] = 1
                # print(mask, mask.shape)
                # c_it_indices = []
                # print(c_it_indices)
                # co_embeddings = self.graph_layer(c_it, n_it, c_embeddings, n_embeddings, prior)    
                co_embeddings = self.graph_layer(c_it,c_embeddings,self.adj)

                prior_out = self.prior_fc(prior)
                # # prior_out = self.relu(prior_out)
                # # prior_out = self.batchnorm(prior_out)

                co_embeddings_with_prior = co_embeddings * prior_out

                co_embeddings_with_prior = self.layernorm(co_embeddings_with_prior)

                co_embeddings_with_prior = self.relu(co_embeddings_with_prior)

                co_embeddings = co_embeddings + co_embeddings_with_prior

                # co_embeddings = self.graph_layer(c_it, c_embeddings)
                visit_embeddings, visit_attention_weights = self.code_attention(co_embeddings,mask, return_score = True)
                batch_attention_weights.append(visit_attention_weights) 
           
                h, c = self.lstm_cell(visit_embeddings,(prev_h,prev_c))
                lstm_outputs.append(h)
                # print(h.shape)
                prev_h, prev_c = h,c
            
            lstm_outputs = torch.stack(lstm_outputs, dim = 0)
            output_attention = self.attention(lstm_outputs,return_score = False)
            output.append(output_attention)

            batch_attention_weights = torch.stack(batch_attention_weights,dim=0)
            attention_weights.append(batch_attention_weights)
        
        output = torch.vstack(output)
        output = self.classifier(output)

        attention_weights = torch.cat(attention_weights, dim = 0)
        if return_attention:
            return output, attention_weights
        else:
            return output
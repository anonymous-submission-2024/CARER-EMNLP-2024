import torch
import torch.nn as nn
from .utils import calculate_laplacian_with_self_loop


class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, device, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self.device = device
        self._num_gru_units = num_gru_units
        # print(self._num_gru_units)
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.feature_size= 32
        self.register_buffer("laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self.weights = nn.Parameter(torch.FloatTensor(self._num_gru_units + self.feature_size, self._output_dim).to(self.device))
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim).to(self.device))
        # self.weights = self.weights.to(self.device)
        # self.biases = self.biases.to(self.device)

        self.reset_parameters()
        self.laplacian = self.laplacian.to(self.device)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes, feature_size = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        # inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape((batch_size, num_nodes, self._num_gru_units))
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape((num_nodes, (self._num_gru_units + feature_size) * batch_size))
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape((num_nodes, self._num_gru_units + feature_size, batch_size))
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        # a_times_concat = a_times_concat.reshape((batch_size * num_nodes, self._num_gru_units + feature_size))
        # print(a_times_concat.shape)
        outputs = torch.zeros((batch_size,num_nodes,self._output_dim)).to(self.device)
        
        # print(outputs.shape)
        # print(a_times_concat.shape)
        # a_times_concat = a_times_concat.to_sparse(
        for i in range(a_times_concat.shape[0]):
            a_times_i_sparse = a_times_concat[i,:,:].to_sparse()
            outputs[i,:,:] = torch.sparse.mm(a_times_i_sparse,self.weights) + self.biases
            # print(output.shape)
            # outputs[i,:,:] = output
            # print(output.shape)
            # outputs[i,:,:] = output
            # outputs.append(output)
        # outputs = torch.vstack(outputs)
        # print(outputs.shape)
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        # print(a_times_concat.device, self.weights.device, self.biases.device)
        # sparse_a_times_concat = a_times_concat.to_sparse()
        # weights = self.weights.unsqueeze(1).repeat(1, K, 1)
        # print(a_times_concat.shape, self.weights.shape, self.biases.shape)
        # outputs = torch.einsum('ijk, kl -> ijl', sparse_a_times_concat, self.weights)
        # print(sparse_a_times_concat.shape)
        # outputs = 
        # outputs = torch.sparse.mm(sparse_a_times_concat, self.weights) + self.biases
        # print(outputs.shape)
        # outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        print(outputs.shape)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGCNCell(nn.Module):
    def __init__(self, adj, device,input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self.device = device
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(self.adj, self.device,self._hidden_dim, self._hidden_dim * 2, bias=1.0)
        self.graph_conv2 = TGCNGraphConvolution(self.adj, self.device, self._hidden_dim, self._hidden_dim)

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN(nn.Module):
    def __init__(self, adj, device, hidden_dim: int = 16, **kwargs):
        super(TGCN, self).__init__()
        self.device= device
        adj = adj.cpu()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.code_num = adj.shape[0]
        # self.register_buffer("adj", torch.FloatTensor(adj.to(self.device)).to(self.device)).to(self.device)
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.code_embedding_size = 32

        self.tgcn_cell = TGCNCell(self.adj, self.device, self._input_dim, self._hidden_dim)
        # self.init_embedding()
        self.embedding = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(self.code_num, self.code_embedding_size))).to(self.device)
        # self.embedding = self.embedding.to(self.device)
    # def init_embedding(self):
    #     self.embedding = nn.Embedding(self.code_num, self.code_embedding_size)
    #     nn.init.xavier_uniform_(self.embedding.weight)
    #     self.embedding.weight = nn.Parameter(self.embedding.weight.to(self.device))
        self.predictor = nn.Linear(64,1).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, visit_lens):
        # print(visit_lens.shape)
        batch_size, seq_len, num_nodes = inputs.shape
       
        # for i in range(seq_len):
            
        # print(inputs)
        inputs = inputs.long()
        inputs = inputs.unsqueeze(-1)
        # print(inputs)
        # for i in seq_len:
        #     inputs[:,i,:] = self.embedding @ inputs[:,i,:]
        inputs = inputs * self.embedding
        # print(inputs.shape)
        # inputs = inputs.mean(dim=2)
        # print(inputs.shape)
        # print("Inputs",inputs[0,0,1,:],inputs.shape)
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)
        output = None
        # all_outputs = []
        outputs = torch.zeros(batch_size,seq_len, num_nodes,self._hidden_dim).to(self.device)
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            outputs[:,i,:,:] = output

        outputs = outputs.mean(dim = 1)
        outputs = outputs.mean(dim = 1)
        print(outputs.shape)
        # batch_size, code_num,hidden_size = output.shape
        # final_output = torch.zeros(batch_size,hidden_size).to(self.device)
        # print(outputs.shape)
        # all_outputs = []
        # for b in range(batch_size):
        #     # print(output[b,:,:].shape)
        #     # print(sample)
        #     sample_output = outputs[b,:visit_lens[b],:]
            # print(sample_output.shape)
            # final_output[b,:] = sample_output
            # print(sample_output.shape)
        # output = output.reshape(batch_size,code_num * hidden_size)
        # print(final_output)
        # final_output
        # output = output.mean(dim = 1)
        # print(output.shape)
        # print(final_output.shape)
        output = self.sigmoid(self.predictor(self.dropout(outputs)))
        # print(output.shape)
        return output

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv, GIN
from torch_geometric.nn import global_add_pool

def get_laplacian_matrix(adj, normalize_L = False):
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

class ChebConvLayer(nn.Module):
    def __init__(self, laplacian,input_dim, output_dim, K, device):
        super(ChebConvLayer, self).__init__()
        self.laplacian = laplacian
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K = K
        self.device = device
        self.weights = nn.Parameter(torch.FloatTensor(K, self.input_dim, self.output_dim)).to(device)
        self.init_weights()

    def forward(self, x):
        # laplacian  = get_laplacian_matrix(adj)
        # print(laplacian.shape)
        laplacian = self.laplacian.unsqueeze(0).repeat(x.shape[0],1,1)
        # print(laplacian)
        cheb_x = []
        x0 = x
        cheb_x.append(x0)

        if self.K > 1:
            x1 = torch.bmm(laplacian, cheb_x[0])
            cheb_x.append(x1)
            for k in range(2, self.K):
                x = 2 * torch.bmm(laplacian, cheb_x[k - 1]) - cheb_x[k - 2]
                cheb_x.append(x)

        chebyshevs = torch.stack(cheb_x, dim=0)
        if chebyshevs.is_sparse:
            chebyshevs = chebyshevs.to_dense()

        output = torch.einsum('hlij,hjk->lik', chebyshevs, self.weights)
        return output

    def init_weights(self):
        nn.init.kaiming_uniform_(self.weights)


# class ChebConvNet(nn.Module):
#     def __init__(self, laplacian,input_dim,output_dim,hidden_dim = 4,K = 1,num_conv_layers = 1,dropout = 0.2, device = 'cuda'):
#         super(ChebConvNet, self).__init__()
#         self.laplacian  =laplacian
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_dim =hidden_dim
#         self.K = K
#         self.num_layers = num_conv_layers
#         self.device = device
#         self.convs = nn.ModuleList()
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=dropout)
#         self.softmax = nn.Softmax(dim=1)
#         assert self.num_layers >= 1, "Number of layers have to be >= 1"

#         if self.num_layers == 1:
#             self.convs.append(
#                 ChebConvLayer(self.laplacian,self.input_dim, self.output_dim, K=self.K, device=self.device).to(self.device))
#         elif self.num_layers >= 2:
#             self.convs.append(
#                 ChebConvLayer(self.laplacian,self.input_dim, self.hidden_dim, K=self.K, device=self.device).to(self.device))
#             for i in range(self.num_layers - 2):
#                 self.convs.append(ChebConvLayer(self.laplacian,self.hidden_dim, self.hidden_dim, device=self.device).to(self.device))
#             self.convs.append(
#                 ChebConvLayer(self.laplacian,self.hidden_dim, self.output_dim, K=self.K, device=self.device).to(self.device))

#     def forward(self, x):
#         # adj = 
#         # laplacian = self.laplacian.unsqueeze(0).repeat(x.shape[0],1,1)
#         for i in range(self.num_layers - 1):
#             x = self.dropout(x)
#             x = self.convs[i](x)
#             x = self.relu(x)


#         x = self.dropout(x)
#         # print('x_before',x)
#         x = self.convs[-1](x)
#         # print('x_after',x)
#         # x = self.softmax(x)
#         # print('x_after_softmax',x)
#         return x


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
        self.relu = nn.LeakyReLU(0.5)
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

        # if self.num_layers == 1:
        #     self.convs.append(GIN(self.input_dim,self.output_dim,num_layers=1).to(self.device))
        # elif self.num_layers >= 2:
        #     self.convs.append(GIN(self.input_dim,self.hidden_dim,num_layers=1).to(self.device))
        #     for i in range(self.num_layers - 2):
        #         self.convs.append(GIN(self.hidden_dim,self.hidden_dim,num_layers=1).to(self.device))
        #     self.convs.append(GIN(self.hidden_dim,self.output_dim,num_layers=1).to(self.device))

    def forward(self, x):
        # adj = 
        # laplacian = self.laplacian.unsqueeze(0).repeat(x.shape[0],1,1)
        edge_index = self.adj.nonzero().t().contiguous()
        row, col = edge_index
        edge_weight = adj[row, col]
        for i in range(self.num_layers - 1):
            x = self.dropout(x)
            x = self.convs[i](x,edge_index)
            x = self.relu(x)

        x = self.dropout(x)
        # print('x_before',x)
        # print(x.shape)
        # print(self.input_dim,self.output_dim)
        x = self.relu(x)
        x = self.convs[-1](x,edge_index)
        # x = global_add_pool(x, batch)
        # print('x_after',x)
        # x = self.softmax(x)
        # print('x_after_softmax',x)
        return x



class LSTM(nn.Module):

    def __init__(self, input_size,  hidden_size = 64, output_size = 1, num_layers  =1, bidirect = True, device = 'cuda'):
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
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=True
        ).to(self.device)
        self.fc = nn.Linear(in_features=self.hidden_size * 2, out_features=self.output_size).to(self.device)
        self.relu = nn.LeakyReLU(0.5)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        h_0 = torch.zeros(self.D * self.num_layers, x.shape[0], self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.D * self.num_layers, x.shape[0], self.hidden_size).to(self.device)
        # Propagate input through LSTM
        # print(x.de)
        # print(x.shape,h_0.shape,c_0.shape)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn[0].view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
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


class DotProductAttention(nn.Module):
    def __init__(self, value_size, attention_size,device):
        super().__init__()
        self.attention_size = attention_size
        self.context = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(attention_size, 1))).to(device)
        self.dense = nn.Linear(value_size, attention_size).to(device)

    def forward(self, x):
        # print(x.shape)
        t = self.dense(x)

        vu = torch.matmul(t, self.context).squeeze()
        score = torch.softmax(vu, dim=-1)
        output = torch.sum(x * torch.unsqueeze(score, dim=-1), dim=-2)
        return output

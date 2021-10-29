import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import math

class GatedGraphConvolution(nn.Module):
    """
    Gated GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features_dim, out_features_dim, lambda_p=0.8, bias=True):
        super(GatedGraphConvolution, self).__init__()
        self.in_features = in_features_dim
        self.out_features = out_features_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_features_dim, out_features_dim))
        self.lambda_p = lambda_p
        self.activation = nn.Tanh()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, dep_adj, latent_adj=None):
        # print("[tlog] text: " + str(text.size()))
        hidden = torch.matmul(text, self.weight)  # B * L * I,  I * O --> B * L * O
        # print("[tlog] hidden: " + str(hidden.size()))
        # sys.exit(0)
        denom = torch.sum(dep_adj, dim=2, keepdim=True) + 1  # B * L * L
        output = torch.matmul(dep_adj, hidden) / denom  # B * L * L , B * L * O --> B * L * O

        # res = True
        # if res:
        #    output = output + text
        dep_output = None
        if self.bias is not None:
            dep_output = output + self.bias
        else:
            dep_output = output

        final_output = dep_output

        if latent_adj is not None and self.lambda_p < 1:
            # hidden = torch.matmul(text, self.weight2)
            denom = torch.sum(latent_adj, dim=2, keepdim=True) + 1  # B * L * L
            output = torch.matmul(latent_adj, hidden) / denom  # B * L * L , B * L * O --> B * L * O

            # res = True
            # if res:
            #    output = output + text
            latent_output = None
            if self.bias is not None:
                latent_output = output + self.bias
            else:
                latent_output = output

            lambda_p = self.lambda_p  # 0.5 # 0.5 for twitter  0.7 for others
            # gate =  (1-lambda_p) * latent_output.sigmoid()
            gate = (1 - lambda_p) * latent_output.sigmoid()

            final_output = (1.0 - gate) * dep_output + gate * latent_output
            # '''
            # final_output = dep_output
            # final_output = latent_output

        # return F.relu(final_output)
        return self.activation(final_output)


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, dim, dropout, num_layers):
        super(GraphConvLayer, self).__init__()

        self.mem_dim = dim
        self.layers = num_layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(dropout)

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim, self.mem_dim)
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(gAxW)
        gcn_ouputs = torch.cat(output_list, dim=2)
        gcn_ouputs = gcn_ouputs + gcn_inputs

        out = self.Linear(gcn_ouputs)
        return out

class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, mem_dim, layers, heads, dropout):
        super(MultiGraphConvLayer, self).__init__()
        #self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(dropout)

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        # query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        # key = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn
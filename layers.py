import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNWithReg(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, k=1, alpha1=0, alpha2=1, alpha3=0.0):
        super(GCNWithReg, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.k = k
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def compute_ABCD(self, x, adj):
        unit = torch.eye(adj.shape[0]).to(adj.device).detach()
        adj_tilda = adj + unit
        self.A = (self.alpha1 + self.alpha2) * unit - self.alpha2 * (1 / torch.sum(adj_tilda, 1, keepdim=True)) * adj_tilda
        self.B = torch.eye(x.shape[1]).to(adj.device).detach()
        self.C = - self.alpha3 * torch.diag((1 / (torch.sum(adj_tilda, 1) + 1e-6)))
        Dx_reverse = torch.diag(torch.sqrt(torch.sum(x * x, 0)))
        self.D = torch.matmul(Dx_reverse, torch.matmul(1-torch.ones(Dx_reverse.shape).to(adj.device).detach()/Dx_reverse.shape[0], Dx_reverse))

    def compute_reg(self, x, adj):
        x0 = x
        self.compute_ABCD(x, adj)
        for _ in range(self.k):
            x = x0 + x - torch.matmul(torch.matmul(self.A, x), self.B) - torch.matmul(torch.matmul(self.C, x), self.D)
        return x

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # e = self._prepare_attentional_mechanism_input(Wh)

        # zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        # attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        # h_prime = torch.matmul(attention, Wh)
        h_prime = self.compute_reg(Wh, adj)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
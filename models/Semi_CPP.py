import time
from models.embedder import embedder_single
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import copy

class EfficientAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Linear(in_channels, key_channels)
        self.queries = nn.Linear(in_channels, key_channels)
        self.values = nn.Linear(in_channels, value_channels)
        self.reprojection = nn.Linear(key_channels, key_channels)

    def forward(self, input_):
        n = input_.size(0)
        h = self.head_count
        # n, _, h, w = input_.size() #n:sample,_ feature, h:head w:batch
        keys = self.keys(input_)
        queries = self.queries(input_)
        values = self.values(input_)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:,i * head_key_channels: (i + 1) * head_key_channels], dim=0)
            query = F.softmax(queries[:,i * head_key_channels: (i + 1) * head_key_channels], dim=1)
            value = values[:,i * head_value_channels: (i + 1) * head_value_channels]
            context = key.transpose(0, 1) @ value
            attended_value = query @context
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)
        return attention

def get_A_r(adj, r):
    adj_label = adj
    for i in range(r - 1):
        adj_label = adj_label @ adj
    return adj_label

def Ncontrast(x_dis, adj_label, tau = 1):
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(0)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention_new(nn.Module):
    def __init__(self, heads, d_model_in, d_model_out, dropout=0.1):
        super().__init__()

        self.d_model = d_model_out
        self.d_k = d_model_out // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model_in, d_model_out)
        self.v_linear = nn.Linear(d_model_in, d_model_out)
        self.k_linear = nn.Linear(d_model_in, d_model_out)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model_out, d_model_out)


    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, self.h, self.d_k)
        q = self.q_linear(q).view(bs, self.h, self.d_k)
        v = self.v_linear(v).view(bs, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 0)
        q = q.transpose(1, 0)
        v = v.transpose(1, 0)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(0, 1).contiguous().view(bs, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, args, d_model, heads, dropout=0.1):
        super().__init__()
        self.args = args
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention_new(heads, d_model,d_model, dropout=dropout)
        self.effectattn =  EfficientAttention(in_channels = d_model, key_channels =d_model, head_count =heads, value_channels = d_model)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm_1(x)
        x = x + self.effectattn(x2)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class CPP_model(nn.Module):
    def __init__(self, arg):
        super(CPP_model, self).__init__()
        self.hid_dim = arg.hid_dim
        self.dropout = arg.random_aug_feature
        self.dropout_att = arg.dropout_att
        self.Trans_layer_num = arg.Trans_layer_num
        self.nheads = arg.nheads
        self.nclass = arg.nclass
        in_channels = arg.ft_size
        self.norm_layer_input = Norm(in_channels)
        self.norm_layer_mid = Norm(arg.MLPdim)
        self.MLPfirst = nn.Linear(in_channels, arg.MLPdim)
        self.norm_input = Norm(in_channels)
        self.Linear_selfC = get_clones(nn.Linear(int(self.hid_dim / self.nheads), self.nclass), self.nheads)
        self.layers = get_clones(EncoderLayer(arg, self.hid_dim, self.nheads, self.dropout_att),self.Trans_layer_num)
        self.norm_trans = Norm(int(self.hid_dim / self.nheads))
        self.layer_singOUT1 = nn.Linear(self.hid_dim, self.nclass)

    def forward(self, x_input):
        x_input = self.norm_layer_input(x_input)
        x = self.MLPfirst(x_input)
        x = F.dropout(x, self.dropout, training=self.training)
        x_dis = get_feature_dis(self.norm_layer_mid(x))
        for i in range(self.Trans_layer_num):
            x = self.layers[i](x)

        D_dim_single = int(self.hid_dim/self.nheads)
        CONN_INDEX = torch.zeros((x.shape[0],self.nclass)).to(x.device)
        for Head_i in range(self.nheads):
            feature_cls_sin = x[:, Head_i*D_dim_single:(Head_i+1)*D_dim_single]
            feature_cls_sin = self.norm_trans(feature_cls_sin)
            Linear_out_one = self.Linear_selfC[Head_i](feature_cls_sin)
            CONN_INDEX += F.softmax(Linear_out_one, dim=1)

        return F.log_softmax(CONN_INDEX, dim=1), x_dis,x

class CPP(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.args.nclass = (self.labels.max() - self.labels.min() + 1).item()
        self.model = CPP_model(self.args).to(self.args.device)
    def training(self):

        features = self.features.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)

        print("Started training...")
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay = self.args.wd)

        if self.idx_train.dtype == torch.bool:
            self.idx_train = torch.where(self.idx_train == 1)[0]
            self.idx_val = torch.where(self.idx_val == 1)[0]
            self.idx_test = torch.where(self.idx_test == 1)[0]

        train_lbls = self.labels[self.idx_train]
        val_lbls = self.labels[self.idx_val]
        test_lbls = self.labels[self.idx_test]

        cnt_wait = 0
        best = 1e-9
        output_acc = 1e-9
        stop_epoch = 0
        start = time.time()
        totalL = []

        adj_label_list = []
        for i in range(2, self.args.order + 1):
            adj_label = get_A_r(graph_org_torch, i)
            adj_label_list.append(adj_label)

        adj_label = adj_label_list[-1]
        del adj_label_list

        index1 = 0

        test_acc_list = []

        for epoch in range(self.args.nb_epochs):
            self.model.train()
            optimiser.zero_grad()
            embeds_tra, x_dis, feature_vector = self.model(features)
            loss_Ncontrast = Ncontrast(x_dis, adj_label, tau=self.args.tau)
            loss = F.cross_entropy(embeds_tra[self.idx_train], train_lbls) + loss_Ncontrast * self.args.beta
            loss.backward()
            optimiser.step()
            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0:
                totalL.append(loss.item())
                self.model.eval()

                embeds, _, feature_vector = self.model(features)
                val_acc = accuracy(embeds[self.idx_val], val_lbls)
                test_acc = accuracy(embeds[self.idx_test], test_lbls)
                print("{:.4f}|".format(test_acc.item()),end="")
                test_acc_list.append(test_acc.item())
                index1 += 1
                if index1 % 10==0:
                    print("")
                    index1 = 0
                # early stop
                stop_epoch = epoch
                if val_acc >= best:
                    best = val_acc
                    output_acc = test_acc.item()
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if cnt_wait == self.args.patience:
                    break
            ################END|Eval|###############

        training_time = time.time() - start
        print("")
        print("\t[Classification] ACC: {:.4f} | stop_epoch: {:}| training_time: {:.4f} ".format(
            output_acc, stop_epoch, training_time))

        return output_acc, training_time, stop_epoch

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

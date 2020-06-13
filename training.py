#!/usr/local/bin/python3
import sys
import csv
import torch
import torch.nn as nn
import numpy as np
from Focal_Loss import *
from torch import optim
import sklearn.metrics as metrics
import numpy as np
from random import randint
from math import floor, ceil

MRNA_INPUT_LEN = 1024
PIRNA_INPUT_LEN = 21
EPOCH = 2000
BATCH_SIZE = 64
THRESHOLD = 0.8
dev = torch.device("cuda:0")


class PyramidNet(nn.Module):
    def __init__(self):
        super(PyramidNet, self).__init__()
        self.conv1 = nn.Conv1d(4, 16, 5, padding=2) # Output size [batch, 16, MRNA_INPUT_LEN]
        self.conv2 = nn.Conv1d(16, 16, 5, padding=2) # Output size [batch, 16, MRNA_INPUT_LEN]
        self.conv3 = nn.Conv1d(16, 16, 5, padding=2) # Output size [batch, 16, MRNA_INPUT_LEN]
        self.conv4 = nn.Conv1d(16, 16, 5, padding=2) # Output size [batch, 16, MRNA_INPUT_LEN]
        self.conv5 = nn.Conv1d(16, 16, 5, padding=2) # Output size [batch, 16, MRNA_INPUT_LEN]

        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool1d(3, 2, 1) # Output size [batch, 16, MRNA_INPUT_LEN / 2]

    def forward(self, mRNA, pyramid_layer):
        # size [batch, 4, MRNA_INPUT_LEN]
        fmap1 = self.conv1(mRNA)
        fact1 = self.act(fmap1)
        fin2 = self.maxpool(fact1)
        # size [batch, 4, MRNA_INPUT_LEN // 2]

        fmap2 = self.conv2(fin2)
        fact2 = self.act(fmap2)
        fin3 = self.maxpool(fact2)
        # size [batch, 4, MRNA_INPUT_LEN // 4]

        fmap3 = self.conv3(fin3)
        fact3 = self.act(fmap3)
        fin4 = self.maxpool(fact3)
        # size [batch, 4, MRNA_INPUT_LEN // 8]

        fmap4 = self.conv4(fin4)
        fact4 = self.act(fmap4)
        fin5 = self.maxpool(fact4)
        # size [batch, 4, MRNA_INPUT_LEN // 16]

        fmap5 = self.conv5(fin5)
        fup4 = fmap4 + nn.Upsample(MRNA_INPUT_LEN // 8)(fmap5)
        fup3 = fmap3 + nn.Upsample(MRNA_INPUT_LEN // 4)(fmap4)
        fup2 = fmap2 + nn.Upsample(MRNA_INPUT_LEN // 2)(fmap4)
        fup1 = fmap1 + nn.Upsample(MRNA_INPUT_LEN)(fmap4)

        return {0: fup1, 1: fup2, 2:fup3, 3: fup4}[pyramid_layer]


# boxnet should apply on segment only
class BoxNet(nn.Module):
    def __init__(self, mapping_layer=3):
        super(BoxNet, self).__init__()
        self.pi_conv1 = nn.Conv1d(4, 8, 5, padding=2)
        self.pi_shrink = nn.Linear(PIRNA_INPUT_LEN, 16)
        self.pi_enlarge = nn.Linear(16, 128)

        self.cls_conv1 = nn.Conv1d(24, 32, 3, padding=1)
        self.cls_conv2 = nn.Conv1d(32, 16, 3, padding=1)
        self.cls_conv3 = nn.Conv1d(16, 2, 3, padding=1)

        self.adj_conv1 = nn.Conv1d(24, 32, 3, padding=1)
        self.adj_conv2 = nn.Conv1d(32, 16, 3, padding=1)
        self.adj_conv3 = nn.Conv1d(16, 2, 3, padding=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, piRNA, pyramid_fmap):
        pi_fmap = self.pi_conv1(piRNA)
        pi_fmap = self.pi_shrink(pi_fmap)
        pi_fmap = self.pi_enlarge(pi_fmap)
        box_in = torch.cat((pi_fmap, pyramid_fmap), dim=1)

        cls_o = self.cls_conv1(box_in)
        cls_o = self.cls_conv2(cls_o)
        cls_o = self.cls_conv3(cls_o)
        cls_o = torch.transpose(cls_o, 1, 2)
        cls_o = self.softmax(cls_o)
        # print(cls_o.shape)

        adj_o = self.adj_conv1(box_in)
        adj_o = self.adj_conv2(adj_o)
        adj_o = self.adj_conv3(adj_o)
        adj_o = torch.transpose(adj_o, 1, 2)

        return cls_o, adj_o

class MPiNet(nn.Module):
    def __init__(self, gamma, alpha):
        super(MPiNet, self).__init__()
        # self.pyramid = PyramidNet().to(dev)
        # self.box = BoxNet().to(dev)
        self.pyramid = PyramidNet()
        self.box = BoxNet()
        self.gamma = gamma
        self.alpha =  alpha
    def forward(self, m_x, pi_x, binding_label, adjust_label):
        out = self.pyramid(m_x, 3)
        pred_cls, pred_adj = self.box(pi_x, out)
        fl = focal_loss(self.alpha, self.gamma, 2)(pred_cls, binding_label)
        sl = nn.functional.smooth_l1_loss(pred_adj, adjust_label)
        return pred_cls, pred_adj, fl + sl


def get_binding():
    # Processing mRNA mapping
    with open("dataset/mRNA_Name_Seq_Mapping.csv") as csvfile:
        rows = list(csv.reader(csvfile))
    mrna2seq = {}
    for i in range(1, len(rows)):
        mrna2seq[rows[i][0]] = rows[i][3]

    # Processing positive and negative samples
    pos_binding = []
    with open("dataset/positive_binding_site.csv") as poscsv:
        pos_rows = list(csv.reader(poscsv))
        pos_rows = pos_rows[1:]
    for p in pos_rows:
        site_location = { "from": int(p[3].split('-')[0]), "to": int(p[3].split('-')[1]) }
        binding = {
            "piRNA_seq": p[0],
            "piRNA_name": p[1],
            "mRNA_name": p[2],
            "mRNA_seq": mrna2seq[p[2]],
            "site_location": site_location,
            "site_seq": mrna2seq[p[2]][site_location["from"]: site_location["to"]]
        }
        pos_binding.append(binding)
    return pos_binding

def inverse_affine_predict(pred_cls, pred_adj, ratio):
    output_cls = np.argmax(pred_cls.cpu().detach().numpy(), axis=2)
    uni, cnt = np.unique(output_cls, return_counts=True)
    print(dict(zip(uni, cnt)))
    output_adj = pred_adj.cpu().detach().numpy()
    output = np.zeros((output_cls.shape[0], MRNA_INPUT_LEN))
    for s in range(output_cls.shape[0]):
        for i, c in enumerate(output_cls[s]):
            if c == 1:
                f = int(output_adj[s, i, 0] * ratio)
                l = int(output_adj[s, i, 1] + ratio)
                e = min(f+l, MRNA_INPUT_LEN // ratio)
                output[s, f: e] = 1
    return output

def seq2onehot(seq, lut):
    encoded = map(lambda el: lut[el], list(seq))
    encoded = np.array(list(encoded))
    return encoded.transpose(1, 0)

def augment_binding(bind, ratio):
    site_l = len(bind["site_seq"])
    pad = MRNA_INPUT_LEN - site_l
    former_space = min(pad, bind["site_location"]["from"])
    n_window = former_space + min(pad, len(bind["mRNA_seq"]) - bind["site_location"]["to"]) - pad
    if n_window < 0:
        return {"validity": False}
    start_i = bind["site_location"]["from"] - former_space + randint(0, n_window)

    eval = np.zeros(MRNA_INPUT_LEN)
    eval[bind["site_location"]["from"] - start_i: bind["site_location"]["to"]] = 1.

    train_cls = np.zeros(MRNA_INPUT_LEN // ratio)
    train_cls[
        floor(bind["site_location"]["from"] / ratio):
        ceil(bind["site_location"]["to"] / ratio)
    ] = 1.

    train_adj = []
    for i in range(MRNA_INPUT_LEN // ratio):
        train_adj.append([
            bind["site_location"]["from"] - (i * ratio + start_i),
            len(bind["site_seq"]) / ratio
        ])
    lut = {
        'A': [1., 0., 0., 0.],
        'T': [0., 1., 0., 0.],
        'C': [0., 0., 1., 0.],
        'G': [0., 0., 0., 1.],
    }
    m = seq2onehot(bind["mRNA_seq"][start_i: start_i + MRNA_INPUT_LEN], lut)
    return {
        "validity": True,
        "eval": eval,
        "train_cls": train_cls,
        "train_adj": train_adj,
        "mRNA": m
    }

def preprocess_label(buffer=16, mapping_layer=3):
    binding = get_binding()
    lut = {
        'A': [1., 0., 0., 0.],
        'T': [0., 1., 0., 0.],
        'C': [0., 0., 1., 0.],
        'G': [0., 0., 0., 1.],
    }
    m = []
    pi = []
    train_cls = []
    train_adj = []
    eval = []
    print(f"total number of sample is {len(binding)}")
    for b in binding[:30000]:
        res = augment_binding(b, 8)
        if not res["validity"]:
            continue
        m.append(res["mRNA"])
        pi.append(seq2onehot(b["piRNA_seq"], lut))
        train_cls.append(res["train_cls"])
        train_adj.append(res["train_adj"])
        eval.append(res["eval"])
    return np.array(m), np.array(pi), np.array(train_cls), np.array(train_adj), np.array(eval)

def split_dataset(valid_ratio, m_x, pi_x, is_binding, adjust):
    assert valid_ratio < 1
    n_training = int(m_x.shape[0] * (1 - valid_ratio))
    index = np.arange(m_x.shape[0])
    np.random.shuffle(index)
    t_i = index[:n_training]
    v_i = index[n_training:]
    return {
        "train": (m_x[t_i], pi_x[t_i], is_binding[t_i], adjust[t_i]),
        "valid": (m_x[v_i], pi_x[v_i], is_binding[v_i], adjust[v_i]),
    }, v_i

def get_next_batch(epoch, batch_size, m, pi, is_binding, adjust):
    for i in range(epoch):
        index = np.arange(m.shape[0])
        np.random.shuffle(index)
        do_valid = True
        for j in range(m.shape[0] // batch_size - 1):
            start = j * batch_size
            end = (j + 1) * batch_size
            yield torch.tensor(m[start: end]), torch.tensor(pi[start: end]), torch.tensor(is_binding[start: end]), torch.tensor(adjust[start: end]), do_valid
            do_valid = False

def train():
    m, pi, train_cls, train_adj, eval = preprocess_label()
    print("m shape", m.shape)
    print("pi shape", pi.shape)
    print("b shape", train_cls.shape)
    print("a shape", train_adj.shape)
    print("e shape", eval.shape)
    tv, v_i = split_dataset(0.1, m, pi, train_cls, train_adj)
    train_m, train_pi, train_binding, train_adjust = tv["train"]
    valid_m, valid_pi, valid_binding, valid_adjust = tv["valid"]
    valid_m = torch.tensor(valid_m).to(dev).float()
    valid_pi = torch.tensor(valid_pi).to(dev).float()
    valid_binding = torch.tensor(valid_binding).to(dev).long()
    valid_adjust = torch.tensor(valid_adjust).to(dev).float()
    net = MPiNet(3.5, 0.05).to(dev)
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    i = 0
    for m, pi, bind_label, adj_label, do_valid in get_next_batch(EPOCH, BATCH_SIZE, train_m, train_pi, train_binding, train_adjust):
        m = m.to(dev).float()
        pi = pi.to(dev).float()
        bind_label = bind_label.to(dev).long()
        adj_label = adj_label.to(dev).float()
        net.train()
        pred_cls, _, loss = net(m, pi, bind_label, adj_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if do_valid:
            net.eval()
            pred_cls, pred_adj, l = net(valid_m, valid_pi, valid_binding, valid_adjust)
            answer = eval[v_i].reshape(-1)
            predict = inverse_affine_predict(pred_cls, pred_adj, 8).reshape(-1)
            f1 = metrics.f1_score(answer, predict)
            acc = metrics.accuracy_score(answer, predict)
            print(f"#{i:3d} f1: {f1} acc: {acc}, , loss: {l}")
            i += 1


if __name__ == "__main__":
    train()

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

        adj_o = self.adj_conv1(box_in)
        adj_o = self.adj_conv2(adj_o)
        adj_o = self.adj_conv3(adj_o)
        adj_o = torch.transpose(adj_o, 1, 2)
        adj_o = self.softmax(adj_o)

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
    assert ratio > 0
    valid = pred_cls[:, :, 0] > THRESHOLD
    output = np.zeros((pred_cls.shape[0], MRNA_INPUT_LEN))
    for s in range(valid.shape[0]):
        for i, c in enumerate(valid[s]):
            if c:
                f = int(pred_adj.cpu().detach().numpy()[s, i, 0] + ratio)
                l = int(pred_adj.cpu().detach().numpy()[s, i, 1] + i * ratio)
                e = max(f+l, MRNA_INPUT_LEN // ratio)
                output[s, f: e] = 1
    return output

def seq2onehot(seq, lut):
    encoded = []
    for el in seq:
        encoded.append(lut[el])
    encoded = np.array(encoded)
    return encoded.transpose(1, 0)

def binding_label(binding, start, end, buff_len, ratio):

    assert ratio > 1
    binding_shifted_start = max(binding["site_location"]["from"] - buff_len, 0)
    binding_shifted_end = min(binding["site_location"]["to"] + buff_len, end)

    if end <= binding_shifted_start or start > binding_shifted_end:
        # [======] binding
        # ----------^------------^----
        #           or
        #                          [======] binding
        # ----------^------------^----
        container = np.empty((MRNA_INPUT_LEN // ratio, 2))
        container[:] = [0., 1.]
        return container
    elif end > binding_shifted_end and binding_shifted_start <= start <= binding_shifted_end:

        #       [======] binding
        # ----------^------------^----
        overlap = np.empty(((end - start) // ratio, 2))
        overlap[:] = [0., 1.]
        overlap[: (binding_shifted_end - start) // ratio] = [1., 0.]
        return np.concatenate((overlap, np.zeros((MRNA_INPUT_LEN // ratio - len(overlap), 2))), axis=0)

    elif end >= binding_shifted_end and start <= binding_shifted_start:

        #              [======] binding
        # ----------^------------^----
        overlap = np.zeros(((end - start) // ratio, 2))
        overlap[:] = [0., 1.]
        overlap[(binding_shifted_start - start) // ratio: (binding_shifted_end - start) // ratio] = [1., 0.]
        return np.concatenate((overlap, np.zeros((MRNA_INPUT_LEN // ratio - len(overlap), 2))), axis=0)
    elif binding_shifted_end > end > binding_shifted_start and start <= binding_shifted_start:

        #                    [======] binding
        # ----------^------------^----
        overlap = np.zeros(((end - start) // ratio, 2))
        overlap[:] = [0., 1.]
        overlap[(binding_shifted_start - start) // ratio:] = [1., 0.]
        return np.concatenate((overlap, np.zeros((MRNA_INPUT_LEN // ratio - len(overlap), 2))), axis=0)
    elif start > binding_shifted_start and end <= binding_shifted_end:

        #       [============] binding
        # ----------^----^------------
        o = np.empty(((end - start) // ratio, 0))
        o[:] = [1., 0.]
        z = np.zeros((MRNA_INPUT_LEN // ratio - len(o), 2), axis=0)
        return np.concatenate((o, z))
    else:
        raise f"Binding relation between piRNA and mRNA is unexpected."

def adjustment_label(binding, start, end, buff_len, ratio):

    adjustment = []
    binding_shifted_start = max(binding["site_location"]["from"] - buff_len, 0)
    binding_shifted_end = min(binding["site_location"]["to"] + buff_len, end)
    binding_len = binding["site_location"]["to"] - binding["site_location"]["from"]
    if end <= binding_shifted_start or start > binding_shifted_end:
        return np.zeros((MRNA_INPUT_LEN // ratio, 2))
    for i in range(start // ratio, end // ratio):
        adjustment.append([binding["site_location"]["from"] - i * ratio, binding_len - ratio])
    z = np.zeros((MRNA_INPUT_LEN // ratio - len(adjustment), 2))
    return np.concatenate((np.array(adjustment), z), axis=0)

def eval_label(binding, start, end, buff_len, ratio):
    b_start = max(binding["site_location"]["from"], 0)
    b_end = min(binding["site_location"]["to"], end)
    output = np.zeros(end - start)
    if end > b_end and b_start <= start <= b_end:
        output[: b_end - start] = 1.
    elif end >= b_end and start <= b_start:
        output[b_start - start: b_end - start] = 1.
    elif b_end > end > b_start and start <= b_start:
        output[b_start - start:] = 1.
    elif start > b_start and end <= b_end:
        output[:] = 1.
    return np.concatenate((output, np.zeros(MRNA_INPUT_LEN - len(output))))

def preprocess_label(buffer=16, mapping_layer=3):
    binding = get_binding()
    lut = {
        'A': [1., 0., 0., 0.],
        'T': [0., 1., 0., 0.],
        'C': [0., 0., 1., 0.],
        'G': [0., 0., 0., 1.],
    }
    m_x = []
    pi_x = []
    is_binding = []
    adjustment = []
    evaluation = []
    for b in binding[:3000]:
        times = len(b["mRNA_seq"]) // MRNA_INPUT_LEN + 1 if len(b["mRNA_seq"]) % MRNA_INPUT_LEN != 0 else 0
        for i in range(times):
            pi_x.append(seq2onehot(b["piRNA_seq"], lut))
            start = i * MRNA_INPUT_LEN
            end = (i + 1) * MRNA_INPUT_LEN
            # Processing input
            if end > len(b["mRNA_seq"]): # Should pad zero here
                valid = seq2onehot(b["mRNA_seq"][start:], lut)
                padding = MRNA_INPUT_LEN - valid.shape[1]
                z = np.zeros((4, padding))
                m_x.append(np.concatenate((valid, z), axis=1))
            else:
                raw_seq = b["mRNA_seq"][start: end]
                m_x.append(seq2onehot(raw_seq, lut))

            # Processing is_binding
            np.zeros(MRNA_INPUT_LEN)
            is_binding.append(binding_label(b, start, end, buffer, 2 ** mapping_layer))
            adjustment.append(adjustment_label(b, start, end, 0, 2 ** mapping_layer))
            evaluation.append(eval_label(b, start, end, buffer, 2 ** mapping_layer))
    return np.array(m_x), np.array(pi_x), np.array(is_binding), np.array(adjustment), np.array(evaluation)

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
            yield torch.tensor(m[start: end]), torch.tensor(pi[start: end]), torch.tensor(is_binding[start: end, :, 0]), torch.tensor(adjust[start: end]), do_valid
            do_valid = False

def train():
    m_x, pi_x, is_binding, adjust, evaluation = preprocess_label()
    tv, v_i = split_dataset(0.05, m_x, pi_x, is_binding, adjust)
    train_m, train_pi, train_binding, train_adjust = tv["train"]
    valid_m, valid_pi, valid_binding, valid_adjust = tv["valid"]
    valid_binding = valid_binding[:, :, 0]
    valid_m = torch.tensor(valid_m).to(dev).float()
    valid_pi = torch.tensor(valid_pi).to(dev).float()
    valid_binding = torch.tensor(valid_binding).to(dev).long()
    valid_adjust = torch.tensor(valid_adjust).to(dev).float()
    net = MPiNet(2, 0.25).to(dev)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
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
            f1 = metrics.f1_score(evaluation[v_i].reshape(-1), inverse_affine_predict(pred_cls, pred_adj, 8).reshape(-1))
            print(f"#{i:3d} evaluate: {f1}, loss: {l}")
            i += 1


if __name__ == "__main__":
    train()

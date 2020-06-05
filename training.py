#!/usr/local/bin/python3
import sys
import csv
import torch
import torch.nn as nn
import numpy as np

MRNA_INPUT_LEN = 1024
MAPPING_LEN = 32

class mRNANet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1D(4, 32, 5, padding=2) # Output size [batch, 16, MRNA_INPUT_LEN]
        self.conv2 = nn.Conv1D(32, 64, 5, padding=2) # Output size [batch, 16, MRNA_INPUT_LEN]
        self.conv3 = nn.Conv1D(64, 64, 5, padding=2) # Output size [batch, 16, MRNA_INPUT_LEN]
        self.conv4 = nn.Conv1D(64, 64, 5, padding=2) # Output size [batch, 16, MRNA_INPUT_LEN]
        self.conv5 = nn.Conv1D(64, 64, 5, padding=2) # Output size [batch, 16, MRNA_INPUT_LEN]

        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool1d(3, 2, 1) # Output size [batch, 16, MRNA_INPUT_LEN / 2]

    def forward(self, mRNA):
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
        fup4 = fmap4 + nn.UpSample(MRNA_INPUT_LEN // 8)(fmap5)
        fup3 = fmap3 + nn.UpSample(MRNA_INPUT_LEN // 4)(fmap4)

        return fup3

# boxnet should apply on segment only
class BoxNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pi_conv = nn.Conv1D(4, 64, 5, padding=2)

        self.cls_conv1 = nn.Conv1D(64, 32, 5, padding=2)
        self.cls_conv2 = nn.Conv1D(32, 1, 5, padding=2)
        self.cls_linear = nn.Linear(MRNA_INPUT_LEN // 4, 2)


        self.adj_conv1 = nn.Conv1D(64, 32, 5, padding=2)
        self.adj_conv2 = nn.Conv1D(32, 1, 5, padding=2)
        self.adj_linear = nn.Linear(MRNA_INPUT_LEN // 4, 2)

    def forward(self, piRNA, pyramid_fmap):
        box_in = self.pi_conv(piRNA) + pyramid_fmap

        cls_o = self.cls_conv1(box_in)
        cls_o = self.cls_conv2(cls_o)
        cls_o = self.cls_linear(cls_o)

        adj_o = self.adj_conv1(box_in)
        adj_o = self.adj_conv2(adj_o)
        adj_o = self.adj_linear(adj_o)

        return cls_o, adj_o


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

def seq2onehot(seq, lut):
    encoded = []
    for el in seq:
        encoded.append(lut[el])
    encoded = np.array(encoded)
    return encoded.transpose(1, 0)

def binding_label(binding, start, end, mapback_len):

    binding_shifted_start = max(binding["site_location"]["from"] - mapback_len, 0)
    binding_shifted_end = min(binding["site_location"]["to"] + mapback_len, end)

    if end <= binding_shifted_start or start > binding_shifted_end:
        # [======] binding
        # ----------^------------^----
        #           or
        #                          [======] binding
        # ----------^------------^----
        container = np.empty((MRNA_INPUT_LEN, 2))
        container[:] = [0., 1.]
        return container
    elif end > binding_shifted_end and binding_shifted_start <= start <= binding_shifted_end:

        #       [======] binding
        # ----------^------------^----
        overlap = np.empty((end - start, 2))
        overlap[:] = [0., 1.]
        overlap[: binding_shifted_end - start] = [1., 0.]
        return np.concatenate((overlap, np.zeros((MRNA_INPUT_LEN - len(overlap), 2))), axis=0)

    elif end >= binding_shifted_end and start <= binding_shifted_start:

        #              [======] binding
        # ----------^------------^----
        overlap = np.zeros((end - start, 2))
        overlap[:] = [0., 1.]
        overlap[binding_shifted_start - start: binding_shifted_end - start] = [1., 0.]
        return np.concatenate((overlap, np.zeros((MRNA_INPUT_LEN - len(overlap), 2))), axis=0)
    elif binding_shifted_end > end > binding_shifted_start and start <= binding_shifted_start:

        #                    [======] binding
        # ----------^------------^----
        overlap = np.zeros((end - start, 2))
        overlap[:] = [0., 1.]
        overlap[binding_shifted_start - start:] = [1., 0.]
        return np.concatenate((overlap, np.zeros((MRNA_INPUT_LEN - len(overlap), 2))), axis=0)
    elif start > binding_shifted_start and end <= binding_shifted_end:

        #       [============] binding
        # ----------^----^------------
        o = np.empty((end - start, 0))
        o[:] = [1., 0.]
        z = np.zeros((MRNA_INPUT_LEN - len(o), 2), axis=0)
        return np.concatenate((o, z))
    else:
        raise f"Binding relation between piRNA and mRNA is unexpected."

def adjustment_label(binding, start, end, mapback_len):

    adjustment = []
    binding_len = binding["site_location"]["to"] - binding["site_location"]["from"]
    for i in range(start, end):
        adjustment.append([binding["site_location"]["from"] - i, binding_len - mapback_len])
    z = np.zeros((MRNA_INPUT_LEN - len(adjustment), 2))
    return np.concatenate((np.array(adjustment), z), axis=0)

def preprocess_label(fmap_len, l2nd_input, buffer=0):
    binding = get_binding()
    lut = {
        'A': [1., 0., 0., 0.],
        'T': [0., 1., 0., 0.],
        'C': [0., 0., 1., 0.],
        'G': [0., 0., 0., 1.],
    }
    x = []
    is_binding = []
    adjustment = []
    for b in binding[:20000]:
        times = len(b["mRNA_seq"]) // MRNA_INPUT_LEN + 1 if len(b["mRNA_seq"]) % MRNA_INPUT_LEN != 0 else 0
        for i in range(times):
            start = i * MRNA_INPUT_LEN
            end = (i + 1) * MRNA_INPUT_LEN
            # Processing input
            if end > len(b["mRNA_seq"]): # Should pad zero here
                valid = seq2onehot(b["mRNA_seq"][start:], lut)
                padding = MRNA_INPUT_LEN - valid.shape[1]
                z = np.zeros((4, padding))
                x.append(np.concatenate((valid, z), axis=1))
            else:
                raw_seq = b["mRNA_seq"][start: end]
                x.append(seq2onehot(raw_seq, lut))

            # Processing is_binding
            np.zeros(MRNA_INPUT_LEN)
            is_binding.append(binding_label(b, start, end, 4))
            adjustment.append(adjustment_label(b, start, end, 4))
    return np.array(x), np.array(is_binding), np.array(adjustment)

def train():
    x, is_binding, adjust = preprocess_label(10, 10)
    print(x.shape, is_binding.shape, adjust.shape)

if __name__ == "__main__":
    train()

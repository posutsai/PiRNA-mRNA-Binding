#!/usr/local/bin/python3
import sys
from itertools import product
import csv
import numpy as np
import torch.nn as nn
import torch
from torch import optim
import random
import sklearn.metrics as metrics


dev = torch.device("cuda:0")
N_GRAM = 1
BATCH_SIZE = 128
EPOCH = 200
PiRNA_L = 21
MRNA_L = 31
def preprocess(pos_path, neg_path):
    pos_data = {"mRNA": [], "piRNA": []}
    neg_data = {"mRNA": [], "piRNA": []}
    with open("dataset/positive_SL31_MFE0_noiso.csv") as csvfile:
        rows = list(csv.reader(csvfile))[1:]
        for r in rows:
            pos_data["piRNA"].append(ngram_encode(r[0]))
            pos_data["mRNA"].append(ngram_encode(r[4]))

    with open("dataset/negative_SL31_MFE0_noiso.csv") as csvfile:
        rows = list(csv.reader(csvfile))[1:]
        for r in rows:
            neg_data["piRNA"].append(ngram_encode(r[1]))
            neg_data["mRNA"].append(ngram_encode(r[3]))
    pos_data["mRNA"] = np.transpose(pos_data["mRNA"], (0, 2, 1))
    pos_data["piRNA"] = np.transpose(pos_data["piRNA"], (0, 2, 1))
    neg_data["mRNA"] = np.transpose(neg_data["mRNA"], (0, 2, 1))
    neg_data["piRNA"] = np.transpose(neg_data["piRNA"], (0, 2, 1))
    return pos_data, neg_data

class FELayer(nn.Module):
    def __init__(self, layer_infos, last_norm=True, norm_type='batch', bias=True):
        super(FELayer, self).__init__()
        self.linear_layers = nn.Sequential()
        for idx, li in enumerate(layer_infos):
            self.linear_layers.add_module(f'linear_{idx}', nn.Linear(li[0], li[1], bias=bias))
            if idx == len(layer_infos) - 1 and not last_norm:
                break
            self.linear_layers.add_module(f'bn_{idx}', nn.LayerNorm(li[1]) if norm_type != 'batch' else nn.BatchNorm1d(li[1]))
            self.linear_layers.add_module(f'relu_{idx}', nn.PReLU())
            if len(li) == 3:
                self.linear_layers.add_module(f'dropout_{idx}', nn.Dropout(li[2]))
    def forward(self, x):
        return self.linear_layers(x)

class SEblock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels//reduction)
        self.relu = nn.PReLU()#(inplace=True)
        self.fc2 = nn.Linear(channels//reduction, channels)
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)

        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        input_x = x
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1)

        return input_x * x

class SELayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, reduction=4, add_residual=False, res_dim=16):
        super(SELayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=(kernel_size//2))
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()#(inplace=True)
        self.se = SEblock(channels=out_channels, reduction=reduction)
        if add_residual:
            self.conv2 = nn.Conv1d(in_channels=res_dim, out_channels=out_channels, kernel_size=1)
            self.bn2 = nn.BatchNorm1d(out_channels)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x, residual=None):
        x = self.conv(x)
        x = self.bn1(x)
        if residual is not None:
            x = x + self.bn1(self.conv2(residual))
        x = self.relu(x)
        x = self.se(x)

        return x

class MPiNet(nn.Module):
    def __init__(self, emb_dim=4, upsample_rate=2, cnn_dim=16):
        super(MPiNet, self).__init__()
        self.pirna_conv = SELayer(emb_dim, cnn_dim, 5, stride=1, reduction=4)
        self.pirna_fe = FELayer([
            [PiRNA_L, PiRNA_L//2, 0.3],
            [PiRNA_L//2, MRNA_L, 0.3],
        ], last_norm=False, norm_type='layer')
        self.cls_conv = SELayer(cnn_dim + 4, 4, 5, stride=1, reduction=4)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=20, nhead=5, dropout=0.15)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        self.cls_input_dim = MRNA_L * 4
        self.cls_layer = FELayer([
            [self.cls_input_dim, self.cls_input_dim//4, 0.3],
            [self.cls_input_dim//4, 2]
        ], last_norm=False, norm_type='batch')

    def _init_hidden_(self):
        return (
            torch.randn(self.rnn_layer*self.rnn_direction, BATCH_SIZE, self.rnn_dim).to(dev),
            torch.randn(self.rnn_layer*self.rnn_direction, BATCH_SIZE, self.rnn_dim).to(dev)
        )
    def forward(self, mrna_x, pirna_x):
        # Originally shape (Batch_size, dim, length)
        # mrna_x = mrna_x.permute(2, 0, 1) (len, batch_size, dim)
        pirna_out = self.pirna_conv(pirna_x)
        pirna_out = self.pirna_fe(pirna_out)
        out = torch.cat((mrna_x, pirna_out), dim=1).permute(2, 0, 1)
        encoded = self.encoder(out).permute(1, 2, 0)
        out = self.cls_conv(encoded)
        out = torch.reshape(out, (BATCH_SIZE, self.cls_input_dim))
        out = self.cls_layer(out)
        return out

def ngram_encode(seq, mode="overlap"):

    def index2list(i):
        r = [0] * (4 ** N_GRAM)
        r[i] = 1
        return r

    if mode not in ("overlap", "side-by-side"):
        raise ValueError("Argument mode should be either 'overlap' or 'side-by-side'")

    lut = { e: index2list(i) for (i, e) in enumerate(product(['A', 'T', 'C', 'G'], repeat=N_GRAM))}
    result = []
    if mode == "overlap":
        for i in range(len(seq) - (N_GRAM - 1)):
            key = tuple(seq[i: i+N_GRAM])
            result.append(lut[key])
    else:
        if len(seq) % N_GRAM != 0:
            raise ValueError("During side-by-side mode the sequence length should be divide by n_gram")

        for i in range(len(seq) // N_GRAM):
            key = tuple(seq[i * N_GRAM: (i+1) * N_GRAM])
            result.append(lut[key])
    return result

def get_next_batch(m, pi, batch_size, train_i):
    batch = 0
    train_size = train_i.shape[0]
    while True:
        if (batch + 1) * BATCH_SIZE > train_size:
            batch = 0
            np.random.shuffle(train_i)
            batch_index = train_i[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE]
        else:
            batch_index = train_i[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE]
            batch += 1
        yield torch.tensor(m[batch_index]).to(dev).float(), torch.tensor(pi[batch_index]).to(dev).float(), batch_index

def train(pos_data, neg_data):
    answers = np.concatenate(
        (np.ones(pos_data["piRNA"].shape[0]), np.zeros(neg_data["piRNA"].shape[0])),
    )
    m = np.concatenate((pos_data["mRNA"], neg_data["mRNA"]), axis=0)
    pi = np.concatenate((pos_data["piRNA"], neg_data["piRNA"]), axis=0)
    index = np.arange(answers.shape[0])
    np.random.shuffle(index)
    ten_fold = answers.shape[0] // 10
    valid_i = index[:ten_fold]
    train_i = index[ten_fold:]
    model = MPiNet().to(dev)
    cse = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total number of trainable {total_params}")
    for e in range(EPOCH):
        model.train()
        train_acc = 0.
        train_b = 0
        for b_m, b_pi, bi in get_next_batch(m, pi, BATCH_SIZE, train_i):
            y_ = model(b_m, b_pi)
            predict = torch.argmax(torch.nn.functional.softmax(y_, 1), 1)
            predict = predict.cpu().detach().numpy()
            train_loss = cse(y_, torch.tensor(answers[bi]).to(dev).long())
            train_acc += metrics.accuracy_score(answers[bi], predict)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_b += 1
            if train_b * BATCH_SIZE > train_i.shape[0]:
                break
        model.eval()
        valid_acc = 0.
        valid_b = 0
        for v_m, v_pi, vi in get_next_batch(m, pi, BATCH_SIZE, valid_i):
            y_ = model(v_m, v_pi)
            predict = torch.argmax(torch.nn.functional.softmax(y_, 1), 1)
            predict = predict.cpu().detach().numpy()
            valid_loss = cse(y_, torch.tensor(answers[vi]).to(dev).long())
            valid_acc += metrics.accuracy_score(answers[vi], predict)
            valid_b += 1
            if valid_b * BATCH_SIZE > valid_i.shape[0]:
                break
        print(f"#{e:3d} th ", end=" ")
        print(f"train_acc is {train_acc / train_b: .3f}, train_loss is {train_loss: .3f}", end="; ")
        print(f"valid_acc is {valid_acc / valid_b: .3f}, valid_loss is {valid_loss: .3f}")

if __name__ == "__main__":
    with open("save_time/processed_dataset.npy", "rb") as f:
        pos_data, neg_data = np.load(f, allow_pickle=True)
    if pos_data["piRNA"].shape[1] != 4 ** N_GRAM:
        print("Repreprocessing dataset ....")
        pos_data, neg_data = preprocess(
            "dataset/positive_SL31_MFE0_noiso.csv",
            "dataset/negative_SL31_MFE0_noiso.csv"
        )
        with open("save_time/processed_dataset.npy", "wb") as f:
            np.save(f, (pos_data, neg_data))
    train(pos_data, neg_data)

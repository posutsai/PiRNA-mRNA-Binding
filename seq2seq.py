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

dev = torch.device("cuda:1")
N_GRAM = 3
BATCH_SIZE_AE = 1024
EPOCH = 20
def ngram_encode(seq):

    def index2list(i):
        r = [0] * (4 ** N_GRAM)
        r[i] = 1
        return r

    lut = { e: index2list(i) for (i, e) in enumerate(product(['A', 'T', 'C', 'G'], repeat=N_GRAM))}
    result = []
    for i in range(len(seq) - (N_GRAM - 1)):
        key = tuple(seq[i: i+N_GRAM])
        result.append(lut[key])
    return result

class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(4 ** N_GRAM, hidden_size, bidirectional=True)
        self.hidden_size = hidden_size

    def forward(self, rna):
        # Assume input rna_seq has shape [batch, ngram_encoding, seq_len]
        rna = rna.permute(2, 0, 1)
        output, hidden = self.rnn(rna)
        encoded = output.view(rna.shape[0], rna.shape[1], 2, self.hidden_size)[-1, :, 0, :]
        return torch.unsqueeze(encoded, 0)

class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.rnn = nn.GRU(hidden_size, 4 ** N_GRAM, bidirectional=True)

    def forward(self, hidden, seq_l):
        state = torch.cat((hidden,) * seq_l)
        output, _ = self.rnn(state)
        seq_l, batch, _ = output.shape
        decoded = output.view(seq_l, batch, 2, 4 ** N_GRAM)[:, :, 0, :]
        return decoded.permute(1, 2, 0)

class AutoEncoderRNA(nn.Module):
    def __init__(self, hidden_size):
        super(AutoEncoderRNA, self).__init__()
        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(hidden_size)

    def forward(self, rna):
        seq_l = rna.shape[-1]
        hidden = self.encoder(rna)
        decoded = self.decoder(hidden, seq_l)
        return decoded, hidden


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

def get_next_batch_ae(m, pi, batch_size):
    train_size = m.shape[0] + pi.shape[0]
    m_i = np.arange(m.shape[0])
    pi_i = np.arange(pi.shape[0])
    np.random.shuffle(m_i)
    np.random.shuffle(pi_i)
    batch_m = 0
    batch_pi = 0
    while True:
        m_or_pi = random.randint(0, 1)
        if m_or_pi > 0:
            if (batch_m + 1) * BATCH_SIZE_AE > m.shape[0]:
                batch_m = 0
                np.random.shuffle(m_i)
                batch_index = m_i[batch_m * BATCH_SIZE_AE: (batch_m + 1) * BATCH_SIZE_AE]
            else:
                batch_index = m_i[batch_m * BATCH_SIZE_AE: (batch_m + 1) * BATCH_SIZE_AE]
                batch_m += 1
            yield m[batch_index]
        else:
            if (batch_pi + 1) * BATCH_SIZE_AE > pi.shape[0]:
                batch_pi = 0
                np.random.shuffle(pi_i)
                batch_index = pi_i[batch_pi * BATCH_SIZE_AE: (batch_pi + 1) * BATCH_SIZE_AE]
            else:
                batch_index = pi_i[batch_pi * BATCH_SIZE_AE: (batch_pi + 1) * BATCH_SIZE_AE]
                batch_pi += 1
            yield pi[batch_index]

def train_AE(pos_data, neg_data):
    pi = np.concatenate((pos_data["piRNA"], neg_data["piRNA"]), axis=0)
    m = np.concatenate((pos_data["mRNA"], neg_data["mRNA"]), axis=0)
    train_pi = pi[2000:]
    train_m = m[2000:]
    valid_pi = pi[:2000]
    valid_m = m[:2000]
    valid_pi = torch.tensor(valid_pi).to(dev).float()
    valid_m = torch.tensor(valid_m).to(dev).float()
    compressor = AutoEncoderRNA(16).to(dev)
    optimizer = optim.SGD(compressor.parameters(), lr=0.01, momentum=0.9)
    # optimizer.zero_grad()
    criterion = nn.NLLLoss()
    for e in range(EPOCH):
        b = 0
        compressor.train()
        for batch in get_next_batch_ae(train_m, train_pi, BATCH_SIZE_AE):
            batch = torch.tensor(batch).to(dev).float()
            decoded, hidden = compressor(batch)
            loss = criterion(decoded, torch.argmax(batch, dim=1))
            loss.backward()
            optimizer.step()
            b += 1
            if b == (pi.shape[0] + m.shape[0]) // BATCH_SIZE_AE:
                break

        # Place validation code here
        compressor.eval()
        loss = 0.
        decoded, _ = compressor(valid_pi)
        loss += criterion(decoded, torch.argmax(valid_pi, dim=1))
        decoded, _ = compressor(valid_m)
        loss += criterion(decoded, torch.argmax(valid_m, dim=1))
        print(f"loss is {loss}")




if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-pp", "--preprocess", default=False, type=bool)
    args = parser.parse_args()
    if args.preprocess:
        pos_data, neg_data = preprocess(
            "dataset/positive_SL31_MFE0_noiso.csv",
            "dataset/negative_SL31_MFE0_noiso.csv"
        )
        with open("save_time/processed_dataset.npy", "wb") as f:
            np.save(f, (pos_data, neg_data))
    else:
        with open("save_time/processed_dataset.npy", "rb") as f:
            pos_data, neg_data = np.load(f, allow_pickle=True)
    train_AE(pos_data, neg_data)


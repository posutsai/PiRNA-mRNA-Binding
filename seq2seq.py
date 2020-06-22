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
import matplotlib.pyplot as plt

# Reference:
# 1. https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

dev = torch.device("cuda:1")
N_GRAM = 1
BATCH_SIZE_AE = 512
EPOCH = 20000
TEACHER_FORCE_RATIO = 0.5
Embedding_LUT = None
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

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="horizontal")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig("weight.png")

class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(4 ** N_GRAM, 4 ** N_GRAM * 4)

    def forward(self, input, hidden):
        # The input of encoder neuron should be permuted beforehand.
        return self.rnn(input, hidden)

    def init_hidden(self):
        return torch.zeros(1, BATCH_SIZE_AE, self.hidden_size, device=dev)
class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(4 ** N_GRAM, 4 ** N_GRAM * 4)
        self.hidden2input = nn.Linear(4 ** N_GRAM * 4, 4 ** N_GRAM)

    def forward(self, input, hidden):
        output, hidden =  self.rnn(input, hidden)
        output = self.hidden2input(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=dev)

class AutoEncoderRNA(nn.Module):
    def __init__(self, hidden_size):
        global Embedding_LUT
        super(AutoEncoderRNA, self).__init__()
        self.criterion = nn.NLLLoss()
        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(hidden_size)
        lut = []
        for i, v in enumerate(product(['A', 'T', 'C', 'G'], repeat=N_GRAM)):
           emb = [0.] * (4 ** N_GRAM)
           emb[i] = 1.
           lut.append(emb)
        Embedding_LUT = nn.Embedding(4 ** N_GRAM, 4 ** N_GRAM)
        Embedding_LUT.weight = nn.Parameter(torch.from_numpy(np.array(lut).astype(np.float32)))
        Embedding_LUT.weight.requires_grad = False
        Embedding_LUT.to(dev)

    def forward(self, ground_truth, tf_rate):
        global Embedding_LUT
        # Setting tf_rate to 0. while evaluating.
        # original shape of ground_truth is (batch, feature_size, seq_l)
        seq_l = ground_truth.shape[2]
        batch = ground_truth.shape[0]
        ground_truth = ground_truth.permute(2, 0, 1)
        # encoding
        enc_hidden = self.encoder.init_hidden()
        enc_collect = torch.zeros(seq_l, batch, self.encoder.hidden_size, device=dev)
        _, enc_hidden = self.encoder(ground_truth, enc_hidden)
        dec_input = torch.zeros(1, batch, 4 ** N_GRAM, device=dev)
        dec_hidden = enc_hidden
        loss = 0.
        output_batch = torch.zeros(seq_l, batch, 4 ** N_GRAM, device=dev)
        if random.random() < tf_rate:
            for di in range(seq_l):
                dec_output, dec_hidden = self.decoder(dec_input, dec_hidden)
                loss += self.criterion(torch.nn.functional.log_softmax(dec_output.squeeze(), 1), torch.argmax(ground_truth[di], 1))
                top_v, top_i = dec_output.topk(1, 2)
                dec_input = Embedding_LUT(torch.squeeze(top_i, 2))
        else:
            # activate teacher-forcing
            for di in range(seq_l):
                dec_output, dec_hidden = self.decoder(dec_input, dec_hidden)
                loss += self.criterion(dec_output.squeeze(), torch.argmax(ground_truth[di], 1))
                dec_input = torch.unsqueeze(ground_truth[di], 0)
                output_batch[di] = dec_output[0]
        return loss, output_batch.permute(1, 2, 0)


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
        # if m_or_pi > 0:
        if False:
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
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.normal_(param.data, 0, 2)

    pi = np.concatenate((pos_data["piRNA"], neg_data["piRNA"]), axis=0)
    m = np.concatenate((pos_data["mRNA"], neg_data["mRNA"]), axis=0)
    train_pi = pi[2000:]
    train_m = m[2000:]
    valid_pi = pi[:2000]
    valid_m = m[:2000]
    valid_pi_tensor = torch.tensor(valid_pi).to(dev).float()
    valid_m_tensor = torch.tensor(valid_m).to(dev).float()
    compressor = AutoEncoderRNA(16).to(dev)
    # compressor.apply(init_weights)
    total_params = sum(p.numel() for p in compressor.parameters())
    print(f"total number of trainable {total_params}")
    optimizer = optim.SGD(compressor.parameters(), lr=0.05, momentum=0.9)
    for e in range(EPOCH):
        b = 0
        compressor.train()
        acc = 0.
        for batch in get_next_batch_ae(train_m, train_pi, BATCH_SIZE_AE):
            batch = torch.tensor(batch).to(dev).float()
            # loss, predict = compressor(batch, TEACHER_FORCE_RATIO)
            loss, predict = compressor(batch, 1.0)
            loss.backward()
            plot_grad_flow(compressor.named_parameters())
            optimizer.step()
            optimizer.zero_grad()
            predict = torch.argmax(torch.nn.functional.softmax(predict, 1), 1)
            predict = predict.cpu().detach().numpy()
            batch = torch.argmax(batch, 1).cpu().detach().numpy()
            acc += metrics.accuracy_score(batch.reshape(-1), predict.reshape(-1))
            b += 1
            # if b == (pi.shape[0] + m.shape[0]) // BATCH_SIZE_AE:
            if b % 100 == 0:
                print(f"acc is {acc / b}, loss is {loss: .3f}")
                break

        # Place validation code here
        compressor.eval()




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


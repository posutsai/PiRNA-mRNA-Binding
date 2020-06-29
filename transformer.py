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
			if idx == len(last_infos) - 1 and not last_norm:
				break
			self.linear_layers.add_module(f'bn_{idx}', nn.LayerNorm(li[1]) if norm_type != 'batch' else nn.BatchNorm1d(li[1]))
			self.linear_layers.add_module(f'relu_{idx}', nn.PReLU())
			if len(li) == 3:
				self.linear_layers.add_module(f'dropout_{idx}', nn.Dropout(li[2]))

    def forward(self, x):
        return self.linear_layers(x)

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

class MPiNet(nn.Module):
    def __init__(self, is_window_mask=True):
        super(MPiNet, self).__init__()
        self.transformer = nn.Transformer(
            d_model=4**N_GRAM,
            nhead=4,
            num_encoder_layers=6,
            num_decoder_layers=6,
            # dropout=0.2
        )
        self.linear1 = nn.Linear(4, 1)
        self.linear2 = nn.Linear(21, 2)
        if is_window_mask:
            self.tgt_mask = self.piRNA_mask_gen(PiRNA_L, 3)
        else:
            self.tgt_mask = self.transformer.generate_square_subsequent_mask(PiRNA_L)
        self.tgt_mask = self.tgt_mask.to(dev)
        # self.tgt_mask.requires_grad = False

    def piRNA_mask_gen(self, seq_l, unmasked):
        mask = (torch.triu(torch.ones(seq_l, seq_l)) == 1).transpose(0, 1)
        submask = torch.triu(
            torch.ones(seq_l - unmasked, seq_l - unmasked) - torch.eye(seq_l - unmasked)
        ) == 1
        mask[unmasked:, :(seq_l - unmasked)] = submask
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, m, pi):
        pi = pi.permute(2, 0, 1)
        m = m.permute(2, 0, 1)
        w = self.transformer(m, pi, tgt_mask=self.tgt_mask)
        w = w.permute(1, 0, 2)
        w = nn.functional.relu(self.linear1(w), 2).permute(0, 2, 1)
        w = torch.flatten(w, start_dim=1)
        w = self.linear2(w)
        return w

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
    valid_i = index[:128]
    train_i = index[2000:]
    model = MPiNet().to(dev)
    cse = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    Total_params = sum(p.numel() for p in model.parameters())
    print(f"total number of trainable {total_params}")
    for e in range(EPOCH):
        model.train()
        train_acc = 0.
        b = 0
        for b_m, b_pi, bi in get_next_batch(m, pi, BATCH_SIZE, train_i):
            y_ = model(b_m, b_pi)
            predict = torch.argmax(torch.nn.functional.softmax(y_, 1), 1)
            predict = predict.cpu().detach().numpy()
            loss = cse(y_, torch.tensor(answers[bi]).to(dev).long())
            train_acc += metrics.accuracy_score(answers[bi], predict)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            b += 1
            if b * BATCH_SIZE > train_i.shape[0]:
                model.eval()
                y_ = model(
                    torch.tensor(m[valid_i]).to(dev).float(),
                    torch.tensor(pi[valid_i]).to(dev).float()
                )
                predict = torch.argmax(torch.nn.functional.softmax(y_, 1), 1)
                predict = predict.cpu().detach().numpy()
                valid_loss = cse(y_, torch.tensor(answers[valid_i]).to(dev).long())
                valid_acc = metrics.accuracy_score(answers[valid_i], predict)

                print(f"train_acc is {train_acc / b: .3f}, loss is {loss: .3f}; valid_acc is {valid_acc:.3f}, valid_loss is {valid_loss:.3f}")
                break

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

import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, all_text, all_label):
        self.all_text = all_text
        self.all_label = all_label
        global max_len, word_2_index, index_2_embedding

    def __getitem__(self, index):
        text = self.all_text[index]
        label = self.all_label[index]
        text_index = [word_2_index[word] for word in text]

        return text_index, label

    def __len__(self):
        return len(self.all_text)
        # return 50000

    def precess_batch(self, datas):  # batch分拣
        batch_max_len = len(datas[-1][0])
        batch_text = []
        batch_label = []
        for data in datas:
            batch_text.append(data[0] + [0] * (batch_max_len - len(data[0])))
            batch_label.append(data[1])

        return torch.tensor(batch_text, dtype=torch.int32), torch.tensor(batch_label)

def count_acc(pre, label):
    res = sum(pre == label) / len(pre)
    return res

def get_data(path, nums=None):
    with open(path, 'r', encoding='UTF-8-sig') as f:
        all_datas = f.read().split('\n')
        all_datas.sort(key=lambda x: len(x))

        all_text = []
        all_label = []
        for datas in all_datas[:nums]:
            data = datas.strip().split('\t')
            if len(data) == 2:
                try:
                    text = data[0]
                    label = int(data[1])

                except Exception as e:
                    continue

                else:
                    all_text.append(text)
                    all_label.append(label)

        return all_text, all_label

def build_word2index(all_text):
    word_to_index = {'<pad>': 0, 'unk': 1}
    for text in all_text:
        for word in text:
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)

    return word_to_index


class Model(nn.Module):
    def __init__(self, corpus_len, embedding_nums, hidden_nums, class_nums):
        super().__init__()
        self.embedding = nn.Embedding(corpus_len, embedding_nums)
        self.LSTM_model = LSTM_model(embedding_nums, hidden_nums)
        self.V = nn.Linear(hidden_nums, class_nums)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        x_emb = self.embedding(x)
        H, t= self.LSTM_model(x_emb)
        pre = self.V(t[0])
        # pre = self.V(H)
        # pre = torch.mean(pre, dim=1)
        if label is not None:
            loss = self.loss_func(pre, label)
            return loss
        else:
            return torch.argmax(pre, dim=1)


class LSTM_model(nn.Module):
    def __init__(self, embedding_nums, hidden_nums):
        super().__init__()
        self.F = nn.Linear(embedding_nums+hidden_nums, hidden_nums)
        self.I = nn.Linear(embedding_nums+hidden_nums, hidden_nums)
        self.C = nn.Linear(embedding_nums+hidden_nums, hidden_nums)
        self.O = nn.Linear(embedding_nums+hidden_nums, hidden_nums)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.hidden_nums = hidden_nums
        self.embedding_nums = embedding_nums

    def forward(self, x, a_pre=None, c_pre=None):
        if a_pre is None:
            a_pre = torch.zeros((x.shape[0], hidden_nums), device=device)
        if c_pre is None:
            c_pre = torch.zeros((x.shape[0], hidden_nums), device=device)

        H = torch.zeros((x.shape[0], x.shape[1], self.hidden_nums), device=device)

        for word_idx in range(x.shape[1]):
            x_a = torch.cat((x[:, word_idx], a_pre), dim=1)
            f = self.F(x_a)
            i = self.I(x_a)
            c = self.C(x_a)
            o = self.O(x_a)

            ft = self.sigmoid(f)
            it = self.sigmoid(i)
            cct = self.tanh(c)
            ot = self.sigmoid(o)

            c_next = ft * c_pre + it * cct
            a_next = self.tanh(c_next) * ot

            c_pre = c_next
            a_pre = a_next

            H[:, word_idx] = a_next

        return H, (a_next, c_next)

if __name__ == '__main__':
    all_text_train, all_label_train = get_data(os.path.join('data', 'train.txt'))
    all_text_dev, all_label_dev = get_data(os.path.join('data', 'dev.txt'))

    assert len(all_text_train) == len(all_label_train), '测试集数据长度不一致'
    assert len(all_text_dev) == len(all_label_dev), '验证集数据长度不一致'

    word_2_index = build_word2index(all_text_train+all_text_dev)
    word_nums = len(word_2_index)
    class_nums = 10
    embedding_nums = 128
    hidden_nums = 100
    batch_size = 50

    epoch = 10
    lr = 0.002
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_dataset = TextDataset(all_text_train, all_label_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_dataset.precess_batch)
    dev_dataset = TextDataset(all_text_dev, all_label_dev)
    dev_dataloader = DataLoader(dev_dataset, batch_size=len(all_text_dev), shuffle=False, collate_fn=dev_dataset.precess_batch)

    model = Model(len(word_2_index), embedding_nums, hidden_nums, class_nums).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for e in range(epoch):
        epoch_loss = 0
        epoch_count = 0
        model.train()
        for batch_text, batch_label in tqdm(train_dataloader):
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
            loss = model(batch_text, batch_label)
            loss.backward()
            opt.step()
            opt.zero_grad()

            epoch_loss += loss
            epoch_count += 1

        model.eval()
        for dev_text, dev_label in tqdm(dev_dataloader):
            dev_text = dev_text.to(device)
            dev_label = dev_label.to(device)
            pre = model(dev_text)
            acc = count_acc(pre, dev_label)

        avg_loss = epoch_loss / epoch_count
        print(f'第{e}个epoch: loss为{avg_loss}, 准确率为{acc}')

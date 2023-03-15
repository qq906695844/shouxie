import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
import math
from torch.nn.utils.rnn import pad_sequence
import pickle
import jieba


def get_data(path, nums=None):
    # english = ["apple", "banana", "orange", "pear", "black", "red", "white", "pink", "green", "blue"]
    # chinese = ["苹果",   "香蕉",     "橙子", "梨",   "黑色", "红色", "白色", "粉红色", "绿色", "蓝色"]
    with open(path, 'r', encoding='utf-8-sig') as f:
        all_datas = f.read().split('\n')[:nums]
    english = []
    chinese = []

    for datas in all_datas:
        data = datas.split('\t')
        if len(data) > 2:
            per_english = data[0].split(' ')
            # per_chinese = jieba.lcut(data[1])
            english.append(per_english)
            # chinese.append(per_chinese)
            chinese.append(data[1])


    return english, chinese

class Model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eng_emb = nn.Embedding(len(params['eng_2_index']), params['embedding_nums'])
        self.chi_emb = nn.Embedding(len(params['chi_2_index']), params['embedding_nums'])
        self.position = Position(params['embedding_nums'])
        self.loss = nn.CrossEntropyLoss()

        self.encoder_blocks = [Encoder_Block(params['embedding_nums'], params['head_nums'], params['ff_nums']) for _ in range(params['block_nums'])]
        # self.encoder_blocks = nn.Sequential(*[Encoder_Block(params['embedding_nums'], params['head_nums'], params['ff_nums']) for _ in range(params['block_nums'])])

        self.decoder_blocks = [Decoder_Block(params['embedding_nums'], params['head_nums'], params['ff_nums']) for _ in range(params['block_nums'])]
        # self.decoder_blocks = nn.Sequential(*[Decoder_Block(params['embedding_nums'], params['head_nums'], params['ff_nums']) for _ in range(params['block_nums'])])
        self.classifier = nn.Linear(params['embedding_nums'], len(params['chi_2_index']))
        self.loss_func = nn.CrossEntropyLoss()
        self.droupout = nn.Dropout(p=0.1)

    def forward(self, batch_eng, batch_chi):
        len_eng = torch.tensor([len(eng) for eng in batch_eng], device=device)
        len_chi = torch.tensor([len(chi) for chi in batch_chi], device=device)

        eng_emb = [self.eng_emb(eng) for eng in batch_eng]
        chi_emb = [self.chi_emb(chi) for chi in batch_chi]

        x = pad_sequence(eng_emb, batch_first=True)
        y = pad_sequence(chi_emb, batch_first=True)
        x = self.position(x)
        y = self.position(y)
        x = self.droupout(x)
        y = self.droupout(y)
        for block in self.encoder_blocks:
            block.to(device)
            x = block(x, len_eng)

        for block in self.decoder_blocks:
            block.to(device)
            y = block(x, y, len_chi, look_ahead_mask=True)

        label = pad_sequence(batch_chi, batch_first=True)[:, 1:]

        pre = self.classifier(y)[:, :-1]
        loss = self.loss_func(pre.reshape(pre.shape[0] * pre.shape[1], -1), label.reshape(-1))
        return loss

    def translate(self, eng):
        eng_index = [self.params['eng_2_index'][chair] for chair in eng]
        eng_emb = self.eng_emb(torch.tensor(eng_index).to(self.params['device'])).unsqueeze(0)
        x = self.position(eng_emb)
        index_2_chi = list(self.params['chi_2_index'])
        for block in self.encoder_blocks:
            block.to(self.params['device'])
            x = block(x)

        res = []
        chair = '<begin>'
        chi_emb = self.chi_emb(torch.tensor([self.params['chi_2_index'][chair]]).to(self.params['device'])).unsqueeze(0)
        out = chi_emb
        while True:
            for block in self.decoder_blocks:
                block.to(self.params['device'])
                out = block(x, out, len_chair=None, look_ahead_mask=False)
            pre_label = int(torch.argmax(self.classifier(out), dim=-1).squeeze(0).tolist()[-1])
            pre = index_2_chi[pre_label]
            if pre == '<end>' or len(res)==20:
                break
            res.append(pre)
            tmp = self.chi_emb(torch.tensor([pre_label]).to(self.params['device'])).unsqueeze(0)
            chi_emb = torch.cat((tmp, chi_emb), dim=1)
            out = chi_emb
        return ''.join(res)


class Encoder_Block(nn.Module):
    def __init__(self, embedding_nums, head_nums, ff_nums):
        super().__init__()
        self.mutil_head_self_att = Mutil_head_self_attention(embedding_nums, head_nums)
        self.add_norm1 = Add_Norm(embedding_nums)
        self.add_norm2 = Add_Norm(embedding_nums)
        self.feed_forward = Feed_forward(embedding_nums, ff_nums)

    def forward(self, batch_x, len_chair=None, look_ahead_mask=False):
        x = self.mutil_head_self_att(batch_x, len_chair, look_ahead_mask)
        a_n_x = self.add_norm1(x)

        a_n_x1 = a_n_x + batch_x  # 残差网络
        ff_x = self.feed_forward(a_n_x1)

        a_n_x2 = self.add_norm2(ff_x)

        res = a_n_x2 + a_n_x1  # 残差网络

        return res


class Decoder_Block(nn.Module):
    def __init__(self, embedding_nums, head_nums, ff_nums):
        super().__init__()
        self.multil_att = Mutil_head_self_attention(embedding_nums, head_nums)
        self.add_norm1 = Add_Norm(embedding_nums)
        self.add_norm2 = Add_Norm(embedding_nums)
        self.add_norm3 = Add_Norm(embedding_nums)
        self.feed_forward = Feed_forward(embedding_nums, ff_nums)
        self.multil_att_en_de = Multil_head_att_en_de(embedding_nums, head_nums)

    def forward(self, batch_eng, batch_chi, len_chair=None, look_ahead_mask=False):
        x1 = self.multil_att(batch_chi, len_chair, look_ahead_mask)
        x_ad1 = self.add_norm1(x1)
        x_ad1 = x1 + x_ad1

        x2 = self.multil_att_en_de(batch_eng, x_ad1)
        x_ad2 = self.add_norm2(x2)
        x_ad2 = x_ad1 + x_ad2

        x3 = self.feed_forward(x_ad2)
        x_ad3 = self.add_norm3(x3)
        x_ad3 = x_ad2 + x_ad3

        return x_ad3

class Mutil_head_self_attention(nn.Module):
    def __init__(self, embedding_nums, head_nums):
        super().__init__()
        self.embedding_nums = embedding_nums
        self.W_O = nn.Linear(embedding_nums, embedding_nums, bias=False)
        if embedding_nums % head_nums == 0:
            head_embedding_nums = int(embedding_nums / head_nums)
        else:
            raise Exception('embedding维度不能整除多头数量')

        self.mutil_head = [Self_Attention(embedding_nums, head_embedding_nums).to(device) for _ in range(head_nums)]

    def forward(self, x, len_chair=None, look_ahead_mask=False):
        out = [self_att(x, len_chair, look_ahead_mask) for self_att in self.mutil_head]
        res = torch.cat(out, dim=-1).to(x.device)
        res = self.W_O(res)
        return res


class Multil_head_att_en_de(nn.Module):  # Encoder_Decoder交互多头注意力
    def __init__(self, embedding_nums, head_nums):
        super().__init__()
        self.embedding_nums = embedding_nums
        self.W_O = nn.Linear(embedding_nums, embedding_nums, bias=False)
        if embedding_nums % head_nums == 0:
            head_embedding_nums = int(embedding_nums / head_nums)
        else:
            raise Exception('embedding维度不能整除多头数量')

        self.mutil_head = [Encoder_Decoder_interactive(embedding_nums, head_embedding_nums).to(device) for _ in range(head_nums)]

    def forward(self, batch_eng, batch_chi):
        out = [self_att(batch_eng, batch_chi) for self_att in self.mutil_head]
        res = torch.cat(out, dim=-1).to(batch_eng.device)
        res = self.W_O(res)
        return res

class Encoder_Decoder_interactive(nn.Module):
    def __init__(self, embedding_nums, head_embedding_nums):
        super().__init__()
        self.embedding_nums = embedding_nums
        self.head_embedding_nums = head_embedding_nums
        self.W_Q = nn.Linear(self.embedding_nums, self.head_embedding_nums, bias=False)
        self.W_K = nn.Linear(self.embedding_nums, self.head_embedding_nums, bias=False)
        self.W_V = nn.Linear(self.embedding_nums, self.head_embedding_nums, bias=False)
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, batch_eng, batch_chi):
        Q = self.W_Q(batch_chi)
        K = self.W_K(batch_eng)
        V = self.W_V(batch_eng)
        score = (Q @ K.transpose(-1, -2)) / math.sqrt(self.head_embedding_nums)
        score = self.soft_max(score)
        res = score @ V
        return res

class Add_Norm(nn.Module):
    def __init__(self, embedding_nums):
        super().__init__()
        self.add = nn.Linear(embedding_nums, embedding_nums)
        self.norm = nn.LayerNorm(embedding_nums)
        self.droupout = nn.Dropout(0.1)

    def forward(self, batch_x, len_chair=None):
        add_x = self.add(batch_x)
        add_x = self.droupout(add_x)
        norm_x = self.norm(add_x)

        return norm_x

class Feed_forward(nn.Module):
    def __init__(self, embeddings_nums, ff_nums):
        super().__init__()
        self.linear1 = nn.Linear(embeddings_nums, ff_nums)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_nums, embeddings_nums)

    def forward(self, batch_x, len_chair=None):
        x = self.linear1(batch_x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

class Self_Attention(nn.Module):
    def __init__(self, embedding_nums, head_embedding_nums):
        super().__init__()
        self.embedding_nums = embedding_nums
        self.head_embedding_nums = head_embedding_nums
        self.W_Q = nn.Linear(self.embedding_nums, self.head_embedding_nums, bias=False)
        self.W_K = nn.Linear(self.embedding_nums, self.head_embedding_nums, bias=False)
        self.W_V = nn.Linear(self.embedding_nums, self.head_embedding_nums, bias=False)
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, batch_x, len_chair=None, look_ahead_mask=False):
        Q = self.W_Q(batch_x)
        K = self.W_K(batch_x)
        V = self.W_V(batch_x)
        score = (Q @ K.transpose(-1, -2)) / math.sqrt(self.head_embedding_nums)

        if len_chair is not None:  # padding mask
            for i in range(score.shape[0]):
                score[i, :, len_chair[i]:] = -1e9

        if look_ahead_mask == True:  # look ahead mask
            for i in range(score.shape[1]):
                score[:, i, i + 1:] = -1e9

        score = self.soft_max(score)
        res = score @ V

        return res



class Position(nn.Module):
    def __init__(self, embedding_nums, max_len=3000):
        super().__init__()
        self.embedding_nums = embedding_nums
        self.pos_codding = torch.zeros(size=(max_len, self.embedding_nums), requires_grad=False)
        t = torch.arange(1, max_len + 1).unsqueeze(1)
        w_i = (1 / 10000 ** (torch.arange(0, self.embedding_nums, 2) / self.embedding_nums)).unsqueeze(0)
        w_i_t = t * w_i
        self.pos_codding[:, ::2] = torch.sin(w_i_t)
        self.pos_codding[:, 1::2] = torch.cos(w_i_t)  # 第一个:是切片，第二个:是取到头

    def forward(self, batch_x):
        batch_pos = self.pos_codding[:batch_x.shape[1]]
        batch_pos = batch_pos.to(batch_x.device)
        return batch_x + batch_pos


class My_Dataset(Dataset):
    def __init__(self, chinese, english, params):
        super().__init__()
        self.chinese = chinese
        self.english = english
        self.params = params

    def __getitem__(self, index):
        eng_idx = [self.params['eng_2_index'][chair] for chair in self.english[index]]
        chi_idx = [self.params['chi_2_index']['<begin>']] + [self.params['chi_2_index'][chair] for chair in self.chinese[index]] + [self.params['chi_2_index']['<end>']]

        return torch.tensor(eng_idx), torch.tensor(chi_idx)

    def __len__(self):
        return len(self.english)

    def process_batch(self, datas):
        batch_eng = []
        batch_chi = []

        for data in datas:
            batch_eng.append(data[0])
            batch_chi.append(data[1])

        return batch_eng, batch_chi

def build_word_2_index(datas):
    word_2_index = {'<pad>': 0, '<unk>': 1, '<begin>': 2, '<end>': 3}
    for data in datas:
        for chair in data:
            word_2_index[chair] = word_2_index.get(chair, len(word_2_index))

    return word_2_index



if __name__ == '__main__':

    english, chinese = get_data(os.path.join('data', 'cmn.txt'), 100)
    assert len(english) == len(chinese), '长度不一致'

    eng_2_index = build_word_2_index(english)
    chi_2_index = build_word_2_index(chinese)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    params = {
        'embedding_nums': 128,
        'head_nums': 2,
        'ff_nums': 128 * 4,
        'eng_2_index': eng_2_index,
        'chi_2_index': chi_2_index,
        'epoch': 400,
        'lr': 0.01,
        'batch_size': 2,
        'block_nums': 3,
        'device': device
    }

    dataset = My_Dataset(chinese, english, params)
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.process_batch)

    model = Model(params).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params['lr'])
    opt_time_step = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.9)

    # with open('param.pkl', 'wb') as f:
    #     pickle.dump(params, f)

    best_loss = 9999
    model.train()
    for i in range(params['epoch']):
        epoch_loss = 0
        epoch_count = 0
        for batch_eng, batch_chi in tqdm(dataloader):
            batch_eng = [eng.to(device) for eng in batch_eng]
            batch_chi = [chi.to(device) for chi in batch_chi]

            loss = model(batch_eng, batch_chi)
            loss.backward()
            opt.step()
            opt.zero_grad()

            epoch_loss += loss
            epoch_count += 1

        opt_time_step.step()
        avg_loss = epoch_loss / epoch_count
        if avg_loss < best_loss:
            best_loss = avg_loss
            # torch.save(model.state_dict(), 'best_model.pt')
            torch.save(model, 'best_model.model')

        print(f'第{i}个epoch: loss为{avg_loss}, 最佳loss为{best_loss}')


    while True:
        input_english = input('请输入英文： ')
        input_english = input_english.split(' ')
        chinese = model.translate(input_english)
        print(chinese)

import jieba
import pandas as pd
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import math
import re
import pickle


def get_data(data_path, stopwords_path):
    texts = pd.read_csv(data_path, encoding='gbk', names=['text'])
    texts = texts['text'].tolist()
    texts_cut = []

    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = f.read().split('\n')

    for text in texts:
        text = re.findall('[\u4e00-\u9fa5]', text)
        text = ''.join(text)
        text_cut = jieba.lcut(text)
        text_cut_no_stopword = [word for word in text_cut if word not in stopwords]
        texts_cut.append(text_cut_no_stopword)

    return texts_cut

def build_word(texts):
    word_2_index = {'<UNK>':0}
    for text in texts:
        for word in text:
            if word not in word_2_index:
                word_2_index[word] = len(word_2_index)
    return word_2_index


def softmax(x):
    max_x = np.max(x, axis=1, keepdims=True)
    x = x - max_x
    exp_x = np.exp(x)
    x_sum = np.sum(exp_x, axis=1, keepdims=True)

    res = exp_x / x_sum
    return res

def sigmoid(x):
    # x = np.clip(x, -700, None)
    res = 1 / (1 + np.exp(-x))
    return res


if __name__ == '__main__':
    text_path = os.path.join('data', '数学原始数据.csv')
    stopword_path = os.path.join('data', 'stopwords.txt')
    text_cut = get_data(text_path, stopword_path)
    word_2_index = build_word(text_cut)
    with open('word_2_index.model', 'wb') as f:
        pickle.dump(word_2_index, f)

    embedding_nums = 200
    epoch = 20
    lr = 0.09
    n_gram = 5
    w1 = np.random.normal(0,1,size=(len(word_2_index),embedding_nums))

    w2 = np.random.normal(0,1,size=(embedding_nums,len(word_2_index)))

    for e in range(epoch):

        epoch_loss = 0
        count = 0
        for text in tqdm(text_cut):
            if len(text) > 1:
                for i, cur_word in enumerate(text):
                    cur_word_idx = word_2_index[cur_word]
                    other_words = text[max(0, i-n_gram): i] + text[i+1: i+1+n_gram]
                    other_words_idx = [word_2_index[other_word] for other_word in other_words]

                    for other_word_idx in other_words_idx:
                        h = w1[cur_word_idx].reshape(1, -1)
                        p = h @ w2[:, other_word_idx].reshape(-1, 1)
                        pre = sigmoid(p)
                        label = np.array([[1]])

                        loss = -np.log(pre)

                        G2 = pre - label
                        delta_w2 = h.T @ G2
                        G1 = G2 @ w2[:, other_word_idx].reshape(-1, 1).T
                        delta_w1 = G1

                        w2[:, other_word_idx] -= lr * delta_w2.reshape(-1)
                        w1[cur_word_idx] -= lr * delta_w1.reshape(-1)
                        epoch_loss += loss
                        count += 1

                    for _ in range(5):
                        no_r_idx = np.random.randint(0, len(word_2_index))
                        while True:
                            if no_r_idx not in other_words_idx:
                                break
                            else:
                                no_r_idx = np.random.randint(0, len(word_2_index))

                        h = w1[cur_word_idx].reshape(1, -1)
                        p = h @ w2[:, no_r_idx].reshape(-1, 1)
                        pre = sigmoid(p)
                        label = np.array([[0]])

                        G2 = pre - label
                        delta_w2 = h.T @ G2
                        w2[:, no_r_idx] -= lr * delta_w2.reshape(-1)


        avg_loss = loss / count * 1e10
        print(f'该epoch的loss为 {float(avg_loss)}')

    np.savetxt('w1.txt', w1)


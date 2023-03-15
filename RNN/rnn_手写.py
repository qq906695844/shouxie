import random
import numpy as np
import os
import gensim
from tqdm import tqdm


def get_data(path, nums=None):
    with open(path, 'r', encoding='utf-8-sig') as f:
        all_datas = f.read().split('\n')
    all_text = []
    for data in all_datas:
        text = [word for word in data]
        all_text.append(text)

    if nums is None:
        return all_text
    else:
        return all_text[:nums]

def get_xy(poetry, word2vec):
    x = poetry[:-1]
    y = poetry[1:]

    x_emb = [word2vec.wv[word] for word in x]
    y_index = [word2vec.wv.key_to_index[word] for word in y]

    return np.array(x_emb), y_index

def softmax(x):
    # x = x - np.max(x)
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    res = x_exp / x_sum
    return res

def label_onehot(y, corpurs_len):
    res = np.zeros((1, corpurs_len))
    res[0, y] = 1
    return res

if __name__ == '__main__':
    path = os.path.join('data', 'poetry_5.txt')
    all_text = get_data(path)


    lr = 0.001
    epoch = 10

    embedding_nums = 128
    hidden_nums = 70


    word2vec = gensim.models.Word2Vec.load('word_2_vec.model')

    word_2_index = word2vec.wv.key_to_index
    index_2_word = word2vec.wv.index_to_key
    corpurs_len = len(index_2_word)

    # 凯明初始化
    W = np.random.normal(0, 2/np.sqrt(embedding_nums), size=(embedding_nums, hidden_nums))
    U = np.random.normal(0, 2/np.sqrt(hidden_nums), size=(hidden_nums, hidden_nums))
    V = np.random.normal(0, 2/np.sqrt(hidden_nums), size=(hidden_nums, corpurs_len))

    W_bias = np.zeros((1, W.shape[1]))
    U_bias = np.zeros((1, U.shape[1]))
    V_bias = np.zeros((1, V.shape[1]))

    for e in range(epoch):
        epoch_loss = 0
        epoch_count = 0
        for poetry in tqdm(all_text):
            x_emb, y_index = get_xy(poetry, word2vec)
            t_pre = np.zeros((1, hidden_nums))

            param = []
            for train_x, train_y in zip(x_emb, y_index):
                h1 = t_pre @ U + U_bias
                h2 = train_x @ W + W_bias

                h_sum = h1 + h2
                t = np.tanh(h_sum)
                o = t @ V + V_bias
                pre = softmax(o)

                label = label_onehot(train_y, corpurs_len)
                loss = -np.sum(label * np.log(pre))
                param.append((pre, label, train_x, train_y, t_pre, t))

                t_pre = t

                epoch_loss += loss
                epoch_count += 1


            dt_pre = 0
            delta_U_sum = 0
            delta_Ub_sum = 0
            delta_W_sum = 0
            delta_Wb_sum = 0
            delta_V_sum = 0
            delta_Vb_sum = 0
            for pre, label, train_x, train_y, t_pre, t in param[::-1]:
                G3 = pre - label
                delta_V = t.T @ G3
                delta_V_bias = G3
                delta_t = G3 @ V.T + dt_pre
                delta_h_sum = delta_t * (1 - t**2)
                delta_h1 = delta_h_sum
                delta_h2 = delta_h_sum
                delta_U = t.T @ delta_h1
                delta_U_bias = delta_h1
                delta_W = train_x.reshape(embedding_nums, 1) @ delta_h2
                delta_W_bias = delta_h2
                dt_pre = delta_h1 @ U.T

                delta_U_sum += delta_U
                delta_Ub_sum += delta_U_bias
                delta_W_sum += delta_W
                delta_Wb_sum += delta_W_bias
                delta_V_sum += delta_V
                delta_Vb_sum += delta_V_bias

            U -= lr * delta_U_sum
            U_bias -= lr * delta_Ub_sum
            W -= lr * delta_W_sum
            W_bias -= lr * delta_Wb_sum
            V -= lr * delta_V_sum
            V_bias -= lr * delta_Vb_sum


        print(epoch_loss/epoch_count)




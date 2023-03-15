import numpy as np
from My_Dataset import Dataset
from sklearn.preprocessing import StandardScaler


dogs = np.array([[8.9, 12], [9, 11], [10, 13], [9.9, 11.2], [12.2, 10.1], [9.8, 13], [8.8, 11.2]],
                dtype=np.float32)  # 0
cats = np.array([[3, 4], [5, 6], [3.5, 5.5], [4.5, 5.1], [3.4, 4.1], [4.1, 5.2], [4.4, 4.4]], dtype=np.float32)  # 1

X = np.vstack([dogs, cats])
y = np.array([0] * 7 + [1] * 7).reshape((-1, 1))
scaler = StandardScaler()
scaler = scaler.fit(X)
X_std = scaler.transform(X)

def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res


w1 = np.random.randn(X.shape[1], 120)
w2 = np.random.randn(120, 1)

b1 = np.random.randn(1)
b2 = np.random.randn(1)

epoch = 1000
lr = 1.0
batch_size = 2

if __name__ == '__main__':

    dataset = Dataset(X_std, y, batch_size, shuffle=True)

    for e in range(epoch):
        loss_sum = [0, 0]
        for batch_x, batch_y in dataset:
            z1 = batch_x @ w1 + b1
            sig_z1 = sigmoid(z1)

            z2 = sig_z1 @ w2 + b2
            batch_hat = sigmoid(z2)

            loss = np.mean(-(batch_y * np.log(batch_hat) + (1-batch_y) * np.log(1-batch_hat)))

            G1 = (batch_hat - batch_y) / batch_x.shape[0]
            delta_w2 = sig_z1.T @ G1
            delta_b2 = np.sum(G1)

            G2 = G1 @ w2.T * (sig_z1 * (1 - sig_z1))
            delta_w1 = batch_x.T @ G2
            delta_b1 = np.sum(G2)

            w1 -= lr * delta_w1
            b1 -= lr * delta_b1
            w2 -= lr * delta_w2
            b2 -= lr * delta_b2


            loss_sum[0] += loss
            loss_sum[1] += 1

        if e % 10 == 0:
            print(f'loss: {loss_sum[0]/loss_sum[1]}')

while True:
    input_1 = np.float64(input('请输入第一维度的特征： '))
    input_2 = np.float64(input('请输入第二维度的特征: '))
    input_x = np.array([input_1, input_2]).reshape(1, -1)

    input_std = scaler.transform(input_x)

    z1 = input_std @ w1 + b1
    sig_z1 = sigmoid(z1)

    z2 = sig_z1 @ w2 + b2
    pre = sigmoid(z2)

    if pre > 0.5:
        print('预测为猫')
    else:
        print('预测为狗')



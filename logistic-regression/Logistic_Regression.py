import numpy as np
import matplotlib.pyplot as plt
import random

# データの生成
data_number = 20
true_b0 = -10
true_b1 = 1
true_b2 = 1

data = []

for _ in range(data_number):
    x1 = np.random.uniform(0,10)
    x2 = np.random.uniform(0,10)
    z = true_b0 + true_b1*x1 + true_b2*x2
    p = 1 / (1 + np.exp(-z))

    if p > 0.5:
        y = 1
    else:
        y = 0

    data.append([x1, x2, y])

input_data = np.array(data)

# 学習の設定
epochs = 500
alpha = 0.005

# パラメータの初期化
b0 = random.uniform(0.1, 5.0)
b1 = random.uniform(0.1, 5.0)
b2 = random.uniform(0.1, 5.0)

for j in range(epochs):

    db0 = 0
    db1 = 0
    db2 = 0

    for i in range(data_number):
        x1 = input_data[i, 0]
        x2 = input_data[i, 1]
        y = input_data[i, 2]
        z = b0 + b1*x1 + b2*x2
        s = 1 / (1 + np.exp(-z))

        # 勾配の計算
        if y == 1:
            db0 = db0 + (1 - s)
            db1 = db1 + (1 - s)*x1
            db2 = db2 + (1 - s)*x2
        else:
            db0 = db0 + (-s)
            db1 = db1 + (-s)*x1
            db2 = db2 + (-s)*x2

    # パラメータの更新
    b0 = b0 + alpha*db0
    b1 = b1 + alpha*db1
    b2 = b2 + alpha*db2

# モデルの可視化
x1_line = np.linspace(0, 10, 100)
x2_line = - (b0 / b2) - (b1 / b2)*x1_line
plt.plot(x1_line, x2_line)

for n in range(data_number):
    x1 = input_data[n, 0]
    x2 = input_data[n, 1]
    y = input_data[n, 2]

    if y == 1:
        plt.scatter(x1, x2, color='red')
    else:
        plt.scatter(x1, x2, color='blue')

plt.show()

# モデルと真の関数の比較
x1_true_line = np.linspace(0, 10, 100)
x2_true_line = - (true_b0 / true_b2) - (true_b1 / true_b2)*x1_true_line
plt.plot(x1_line, x2_line, label="model", color='red')
plt.plot(x1_true_line, x2_true_line, label="true", color="blue")

for n in range(data_number):
    x1 = input_data[n, 0]
    x2 = input_data[n, 1]
    y = input_data[n, 2]

    if y == 1:
        plt.scatter(x1, x2, color='red')
    else:
        plt.scatter(x1, x2, color='blue')

plt.legend()
plt.show()

# 学習したパラメータと真のパラメータの比較
print(b0, b1, b2)
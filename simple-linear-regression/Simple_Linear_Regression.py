import numpy as np
import random
import matplotlib.pyplot as plt

# データの生成
data = []
a = 2.5
b = 0.5
for i in range(20):
    x = random.uniform(0, 35)
    noise = random.uniform(-2.5, 2.5)
    y = a + b*x + noise
    data.append([x,y])

input_data = np.array(data)
data_number = input_data.shape[0]

# 学習の設定
epochs = 50
alpha = 0.00002

# パラメータの初期化
w0 = random.uniform(0.01, 0.2)
w1 = random.uniform(0.01, 0.2)

loss_history = []

for j in range(epochs):
    dw0 = 0
    dw1 = 0
    loss = 0

    for i in range(data_number):
        # lossの計算
        x = input_data[i,0]
        y = input_data[i,1]
        y_pred = w0 + w1*x
        loss += (y_pred - y)**2

        # 勾配の計算
        dw0 = dw0 + 2*w0 + 2*w1*x - 2*y
        dw1 = dw1 + x*(2*w1*x + 2*w0 - 2*y)

    loss_history.append(loss)

    # パラメータの更新
    w0 = w0 - alpha*dw0
    w1 = w1 - alpha*dw1

# モデルの可視化
x = np.linspace(0,35,100)
model_y = w0 + w1*x
plt.plot(x, model_y, label="model", color="red")
for u in range(data_number):
    plt.scatter(input_data[u,0],input_data[u,1])
plt.show()

# 学習の進捗
plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Epoch vs Loss")
plt.show()

# モデルと真の関数の比較
x = np.linspace(0,35,100)
true_y = a + b*x
plt.plot(x, model_y, label="model", color="red")
plt.plot(x, true_y, label="true", color="blue")
for u in range(data_number):
    plt.scatter(input_data[u,0],input_data[u,1])
plt.legend()
plt.show()

# 学習したパラメータと真のパラメータの比較
print(a, b, w0, w1)
import numpy as np
import matplotlib.pyplot as plt

# データの生成
np.random.seed(42)
data_number = 100
input_data = np.random.uniform(1, 10, (data_number, 2))
labels = np.zeros(data_number)

for i in range(data_number):
    x1 = input_data[i, 0]
    x2 = input_data[i, 1]
    if (x1 > 5.5 and x2 > 5.5) or (x1 < 5.5 and x2 < 5.5):
        labels[i] = 1
    else:
        labels[i] = 0

for i in range(data_number):
    if np.random.rand() < 0.02:
        labels[i] = 1 - labels[i]

# データの可視化
plt.figure(figsize=(6, 6))
plt.scatter(input_data[labels==0, 0], input_data[labels==0, 1], label="Class 0", marker="o")
plt.scatter(input_data[labels==1, 0], input_data[labels==1, 1], label="Class 1", marker="x")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()

def S(z):
    return 1 / (1 + np.exp(-z))

# パラメータの初期化
init_weight = np.random.rand(3, 3)
w111 = init_weight[0, 0]
w112 = init_weight[0, 1]
w121 = init_weight[0, 2]
w122 = init_weight[1, 0]
b11 = init_weight[1, 1]
b12 = init_weight[1, 2]
w211 = init_weight[2, 0]
w212 = init_weight[2, 1]
b2 = init_weight[2, 2]

# 学習の設定
epochs = 1000
alpha = 0.01

loss_history = []

for j in range(epochs):
    total_loss = 0

    for i in range(data_number):
        x1 = input_data[i, 0]
        x2 = input_data[i, 1]
        label = labels[i]

        Z11 = b11 + w111*x1 + w112*x2
        Z12 = b12 + w121*x1 + w122*x2
        O11 = S(Z11)
        O12 = S(Z12)
        Z21 = b2 + w211*O11 + w212*O12
        O21 = S(Z21)

        # 勾配の計算
        dw111 = 2*(O21 - label)*S(Z21)*(1 - S(Z21))*w211*S(Z11)*(1 - S(Z11))*x1
        dw112 = 2*(O21 - label)*S(Z21)*(1 - S(Z21))*w211*S(Z11)*(1 - S(Z11))*x2
        db11  = 2*(O21 - label)*S(Z21)*(1 - S(Z21))*w211*S(Z11)*(1 - S(Z11))
        db12  = 2*(O21 - label)*S(Z21)*(1 - S(Z21))*w212*S(Z12)*(1 - S(Z12))
        dw121 = 2*(O21 - label)*S(Z21)*(1 - S(Z21))*w212*S(Z12)*(1 - S(Z12))*x1
        dw122 = 2*(O21 - label)*S(Z21)*(1 - S(Z21))*w212*S(Z12)*(1 - S(Z12))*x2
        dw211 = 2*(O21 - label)*S(Z21)*(1 - S(Z21))*O11
        dw212 = 2*(O21 - label)*S(Z21)*(1 - S(Z21))*O12
        db2   = 2*(O21 - label)*S(Z21)*(1 - S(Z21))

        # パラメータの更新
        w111 -= alpha * dw111
        w112 -= alpha * dw112
        b11   -= alpha * db11
        b12   -= alpha * db12
        w121 -= alpha * dw121
        w122 -= alpha * dw122
        w211 -= alpha * dw211
        w212 -= alpha * dw212
        b2   -= alpha * db2

        total_loss += (O21 - label)**2

    loss_history.append(total_loss / data_number)

# モデルの可視化
x1_range = np.linspace(1, 10, 200)
x2_range = np.linspace(1, 10, 200)
X1, X2 = np.meshgrid(x1_range, x2_range)

Z11_grid = b11 + w111*X1 + w112*X2
Z12_grid = b12 + w121*X1 + w122*X2
O11_grid = S(Z11_grid); O12_grid = S(Z12_grid)
Z21_grid = b2 + w211*O11_grid + w212*O12_grid
O21_grid = S(Z21_grid)

plt.figure(figsize=(6, 6))
cp = plt.contourf(X1, X2, O21_grid, levels=50, cmap="RdBu_r", alpha=0.8)
plt.colorbar(cp, label="Class 1")
plt.contour(X1, X2, O21_grid, levels=[0.5], colors="black", linewidths=2)
plt.scatter(input_data[labels==0, 0], input_data[labels==0, 1], label="Class 0", marker="o", edgecolors="white", s=50)
plt.scatter(input_data[labels==1, 0], input_data[labels==1, 1], label="Class 1", marker="x", color="yellow", s=70)
plt.xlabel("x1");
plt.ylabel("x2")
plt.legend();
plt.grid(True)
plt.show()

# 学習の進捗
plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Epoch vs Loss")
plt.grid(True)
plt.show()

# モデルと真の境界線の比較
plt.figure(figsize=(6, 6))

cp = plt.contourf(X1, X2, O21_grid, levels=50, cmap="RdBu_r", alpha=0.8)
plt.colorbar(cp, label="Class 1")
plt.contour(X1, X2, O21_grid, levels=[0.5], colors="black", linewidths=2)

plt.axvline(x=5.5, color="lime", linewidth=2, linestyle="--", label="true boundary")
plt.axhline(y=5.5, color="lime", linewidth=2, linestyle="--")
plt.scatter(input_data[labels==0, 0], input_data[labels==0, 1], marker="o", edgecolors="white", s=50, label="Class 0")
plt.scatter(input_data[labels==1, 0], input_data[labels==1, 1], marker="x", color="yellow", s=70, label="Class 1")

plt.title("True vs Model boundary")
plt.xlabel("x1"); plt.ylabel("x2")
plt.legend(); plt.grid(True)
plt.xlim(1, 10); plt.ylim(1, 10)
plt.show()
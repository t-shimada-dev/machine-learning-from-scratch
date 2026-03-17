import numpy as np
import matplotlib.pyplot as plt

# データの生成
np.random.seed(42)
data_number = 150

X0 = np.random.randn(50, 2) * 0.8 + np.array([1, 1])
X1 = np.random.randn(50, 2) * 0.8 + np.array([5, 5])
X2 = np.random.randn(50, 2) * 0.8 + np.array([1, 5])

X = np.vstack([X0, X1, X2])
y = np.array([0]*50 + [1]*50 + [2]*50)

# データの可視化
plt.figure(figsize=(6, 6))
for cls, label in zip([0, 1, 2], ["Class 0", "Class 1", "Class 2"]):
    mask = y == cls
    plt.scatter(X[mask, 0], X[mask, 1], label=label)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()

# 決定木の実装
# Gini不純度の計算
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

# 最も多いクラスを返す関数
def most_common(y):
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmax(counts)]

# 情報利得の計算
def information_gain(X_col, y, threshold):
    parent_gini = gini(y)

    left_mask = X_col <= threshold
    left_y  = y[left_mask]
    right_y = y[~left_mask]

    if len(left_y) == 0 or len(right_y) == 0:
        return 0

    n = len(y)
    child_gini = (len(left_y) / n) * gini(left_y) + (len(right_y) / n) * gini(right_y)

    return parent_gini - child_gini

# 最適な分割を見つける関数
def best_split(X, y):
  best_gain = -1
  best_feature = None
  best_threshold = None

  for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            gain = information_gain(X[:, feature], y, threshold)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            best_threshold = threshold

  return best_feature, best_threshold

# 決定木の構築
def build_tree(X, y, depth, max_depth=5, min_samples_split=2):
    n_samples = len(y)

    if depth >= max_depth or n_samples < min_samples_split or len(np.unique(y)) == 1:
        return {"leaf": True, "value": most_common(y)}

    best_feature, best_threshold = best_split(X, y)

    if best_feature is None:
        return {"leaf": True, "value": most_common(y)}

    left_mask = X[:, best_feature] <= best_threshold
    left_tree  = build_tree(X[left_mask],  y[left_mask],  depth + 1, max_depth, min_samples_split)
    right_tree = build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth, min_samples_split)

    return {
        "leaf": False,
        "feature": best_feature,
        "threshold": best_threshold,
        "left": left_tree,
        "right": right_tree
    }

# 予測関数
def predict_one(x, node):
    if node["leaf"]:
        return node["value"]
    if x[node["feature"]] <= node["threshold"]:
        return predict_one(x, node["left"])
    else:
        return predict_one(x, node["right"])

# 複数のサンプルに対する予測関数
def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])

# 決定木の構築
tree = build_tree(X, y, depth=0, max_depth=5)

# 決定境界の可視化
x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
x2_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

grid = np.c_[X1_grid.ravel(), X2_grid.ravel()]
Z = predict(grid, tree).reshape(X1_grid.shape)

plt.figure(figsize=(6, 6))
plt.contourf(X1_grid, X2_grid, Z, alpha=0.4, cmap="Set1")
for cls, label in zip([0, 1, 2], ["Class 0", "Class 1", "Class 2"]):
    mask = y == cls
    plt.scatter(X[mask, 0], X[mask, 1], label=label)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.title("Decision Tree Boundary")
plt.show()

# モデルの評価
y_pred = predict(X, tree)
accuracy = np.sum(y_pred == y) / len(y)
print(f"Accuracy: {accuracy:.4f}")

# max_depthを変化させたときの精度の変化をプロット
accuracies = []
depths = range(1, 11)

for max_depth in depths:
    tree = build_tree(X, y, depth=0, max_depth=max_depth)
    y_pred = predict(X, tree)
    accuracy = np.sum(y_pred == y) / len(y)
    accuracies.append(accuracy)

plt.figure(figsize=(6, 4))
plt.plot(depths, accuracies, marker="o")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("max_depth vs Accuracy")
plt.grid(True)
plt.show()

# 混同行列の作成
n_classes = 3
cm = np.zeros((n_classes, n_classes), dtype=int)
for true, pred in zip(y, y_pred):
    cm[true][pred] += 1

# 混同行列の可視化
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
for i in range(n_classes):
    for j in range(n_classes):
        plt.text(j, i, cm[i][j], ha="center", va="center", color="black")
plt.xticks([0, 1, 2], ["Class 0", "Class 1", "Class 2"])
plt.yticks([0, 1, 2], ["Class 0", "Class 1", "Class 2"])
plt.show()
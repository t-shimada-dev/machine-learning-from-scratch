import numpy as np
import matplotlib.pyplot as plt

# スロットの真の成功確率
np.random.seed(42)
true_probs = [0.2, 0.35, 0.5, 0.65, 0.8]
n_bandits = len(true_probs)
n_rounds = 1000

# UCBアルゴリズムの実装
def ucb(n_bandits, n_rounds, true_probs):
    N = np.zeros(n_bandits)
    R = np.zeros(n_bandits)
    choices = []
    rewards = []
    regrets = []

    best_prob = max(true_probs)

    for i in range(n_bandits):
        reward = np.random.binomial(1, true_probs[i])
        N[i] += 1
        R[i] += reward

    for n in range(n_bandits, n_rounds):
        ucb_values = R / N + np.sqrt(3/2 * np.log(n) / N)
        chosen = np.argmax(ucb_values)
        reward = np.random.binomial(1, true_probs[chosen])
        N[chosen] += 1
        R[chosen] += reward
        choices.append(chosen)
        rewards.append(reward)
        regrets.append(best_prob - true_probs[chosen])

    return choices, rewards, N, R, regrets

choices, rewards, N, R, regrets = ucb(n_bandits, n_rounds, true_probs)

# 選択回数の可視化
plt.bar(range(n_bandits), N, color='steelblue')
plt.xticks(range(n_bandits), [f'Slot {i}\n(p={true_probs[i]})' for i in range(n_bandits)])
plt.xlabel("Slot")
plt.ylabel("Number of Selections")
plt.title("UCB: Number of Selections per Slot")
plt.show()

# 報酬の可視化
cumulative_rewards = np.cumsum(rewards)
plt.plot(cumulative_rewards)
plt.xlabel("Round")
plt.ylabel("Cumulative Reward")
plt.title("UCB: Cumulative Reward")
plt.show()

# regretの可視化
cumulative_regret = np.cumsum(regrets)
plt.plot(cumulative_regret)
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("UCB: Cumulative Regret")
plt.show()
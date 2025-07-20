import numpy as np
import matplotlib.pyplot as plt

class AxiomBandit:
    def __init__(self, K, prior_mean=0.0, prior_var=1.0):
        self.K = K
        self.means = np.full(K, prior_mean, dtype=float)
        self.vars = np.full(K, prior_var, dtype=float)
        self.counts = np.zeros(K, dtype=int)
        self.history = {"actions": [], "rewards": [], "beliefs": [], "uncertainty": [], "pulls": []}
        self.internal_goal = 0.67
        self.all_rewards = []

    def select_action(self, timestep):
        # 🔁 Enforced periodic reevaluation
        if timestep % 20 == 0:
            unexplored = np.where(self.counts == 0)[0]
            if unexplored.size > 0:
                return np.random.choice(unexplored)
            least_sampled = np.argmin(self.counts)
            return least_sampled

        # 🤖 Expected Free Energy minimization
        epistemic = self.vars
        #pragmatic = self.means
        pragmatic = (self.internal_goal - np.mean(self.all_rewards))*self.means
        bonus = 0.05 / (np.sqrt(self.counts + 1))  # Soft exploration buffer
        efe = -pragmatic + epistemic + bonus
        return np.argmin(efe)

    def update(self, action, reward):
        self.counts[action] += 1
        lr = 1 / (np.sqrt(self.counts[action]))  # More cautious update

        # ⬆️ Belief update
        self.means[action] += lr * (reward - self.means[action])

        # ⬇️ Uncertainty decay (slower to preserve epistemic motivation)
        self.vars[action] *= (1 - 0.1 * lr)
        self.vars[action] = max(self.vars[action], 1e-3)

        # 📊 Log history for visualization
        self.history["actions"].append(action)
        self.history["rewards"].append(reward)
        self.history["beliefs"].append(self.means.copy())
        self.history["uncertainty"].append(self.vars.copy())
        self.history["pulls"].append(self.counts.copy())

    def plot_learning(self, true_means):
        timesteps = len(self.history["beliefs"])
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # 🔍 Estimated reward beliefs
        for k in range(self.K):
            beliefs = [b[k] for b in self.history["beliefs"]]
            axs[0].plot(beliefs, label=f'Arm {k} (true: {true_means[k]:.2f})')
        axs[0].set_title("Estimated Rewards")
        axs[0].set_xlabel("Timestep")
        axs[0].set_ylabel("Mean Estimate")
        axs[0].legend()
        axs[0].grid(True)

        # ⚖️ Uncertainty
        for k in range(self.K):
            uncertainties = [u[k] for u in self.history["uncertainty"]]
            axs[1].plot(uncertainties, label=f'Arm {k}')
        axs[1].set_title("Uncertainty (Variance)")
        axs[1].set_xlabel("Timestep")
        axs[1].set_ylabel("Variance")
        axs[1].legend()
        axs[1].grid(True)

        # 📈 Pull Frequencies
        final_counts = self.history["pulls"][-1]
        axs[2].bar(range(self.K), final_counts)
        axs[2].set_title("Arm Pull Frequency")
        axs[2].set_xlabel("Arm")
        axs[2].set_ylabel("Times Pulled")
        axs[2].grid(True)
        

        plt.tight_layout()
        plt.show()

# 🚀 Run Simulation
np.random.seed(37)
K = 3
true_means = [0.1, 0.3, 0.8]  # Blue, Orange, Green

bandit = AxiomBandit(K=K)
T = 10000

for t in range(T):
    action = bandit.select_action(t)
    reward = np.random.binomial(1, true_means[action])
    bandit.all_rewards.append(reward)
    bandit.update(action, reward)

print(np.mean(bandit.all_rewards))

# Verifications
prob = []
for i in range(3):
    prob.append(bandit.history["pulls"][-1][i]/T)

print(np.sum(np.array(prob)*np.array(true_means)))

bandit.plot_learning(true_means)

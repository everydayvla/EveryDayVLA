import numpy as np

class ThresholdBanditTuner:
    def __init__(self, thresholds, epsilon=0.1):
        self.thresholds = thresholds
        self.n_arms = len(thresholds)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms, dtype=float)
        self.epsilon = epsilon
        self.total = 0

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.values))

    def update(self, arm, reward):
        self.total += 1
        self.counts[arm] += 1
        # incremental mean
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import trange

EMPTY, AGENT, MESS, CLEAN = range(4)

class RewardHackEnv:
    # init environment state and RNG
    def __init__(self, size=5, n_messes=4, max_steps=80, seed=None):
        self.size = size
        self.n_messes = n_messes
        self.max_steps = max_steps
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.reset()

    # reset episode state
    def reset(self):
        self.agent_pos = (0, 0)
        self.messes = set()
        self.steps = 0
        while len(self.messes) < self.n_messes:
            pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if pos != self.agent_pos:
                self.messes.add(pos)
        self.cleaned = set()
        return self.get_obs()

    # encode observable state
    def get_obs(self):
        return (self.agent_pos, tuple(sorted(self.messes)))

    # advance environment dynamics
    def step(self, action, hack=False):
        self.steps += 1
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[action]
        new_x = np.clip(self.agent_pos[0] + dx, 0, self.size - 1)
        new_y = np.clip(self.agent_pos[1] + dy, 0, self.size - 1)
        self.agent_pos = (int(new_x), int(new_y))
        true_reward = -0.02
        if self.agent_pos in self.messes:
            self.messes.remove(self.agent_pos)
            self.cleaned.add(self.agent_pos)
            true_reward += 1.0
        observed_reward = true_reward
        if hack:
            observed_reward += 1.3
        done = (len(self.messes) == 0) or (self.steps >= self.max_steps)
        return self.get_obs(), observed_reward, true_reward, done


# train tabular Q-learning agent
def train(n_episodes=2500, alpha=0.12, gamma=0.92, epsilon=0.25,
          epsilon_decay=0.9993, hack_prob=0.0, seed=42):

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    env = RewardHackEnv()
    Q = defaultdict(float)
    obs_history = []
    true_history = []
    eps = epsilon

    for _ in trange(n_episodes, desc=f"hack={hack_prob:.2f}"):
        state = env.reset()
        done = False
        ep_obs_reward = 0.0
        ep_true_reward = 0.0

        while not done:
            if random.random() < eps:
                action = random.randint(0, 3)
            else:
                action = np.argmax([Q[(state, a)] for a in range(4)])

            hack_this_step = random.random() < hack_prob
            next_state, obs_r, true_r, done = env.step(action, hack=hack_this_step)

            best_next = max(Q[(next_state, a)] for a in range(4))
            Q[(state, action)] += alpha * (obs_r + gamma * best_next - Q[(state, action)])

            state = next_state
            ep_obs_reward += obs_r
            ep_true_reward += true_r

        obs_history.append(ep_obs_reward)
        true_history.append(ep_true_reward)
        eps = max(0.02, eps * epsilon_decay)

    return obs_history, true_history


# run experiment and visualize outcomes
if __name__ == "__main__":
    normal_obs, normal_true = train(hack_prob=0.0, seed=42)
    hacked_obs, hacked_true = train(hack_prob=0.45, seed=42)

    def smooth(x, window=60):
        return np.convolve(x, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(13, 6.5))
    plt.plot(smooth(normal_true), lw=2.4, label="Normal true reward")
    plt.plot(smooth(hacked_true), lw=1.8, alpha=0.7, label="Hacked true reward")
    plt.plot(smooth(hacked_obs), ls="--", lw=2.2, label="Hacked perceived reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.grid(alpha=0.35)
    plt.tight_layout()
    plt.show()

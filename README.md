# alignment-failure-modes
Visual demonstration of reward hacking in reinforcement learning agents using a tabular Q-learning setup.
# Reward Hacking Demonstration

Reward hacking occurs when a reinforcement learning agent exploits weaknesses in a reward signal rather than accomplishing the intended task. The goal of the environment in this repository is to illustrate how an agent can learn policies that maximize perceived reward while reducing or even harming real task performance.

The environment models a simple cleaning task in a grid world. An agent navigates the grid and cleans messes to receive reward. Under normal conditions, reward reflects genuine task progress. Under hacked conditions, the agent occasionally receives artificial reward that does not correspond to real cleaning actions. Learning is driven entirely by the perceived reward rather than the true environmental outcome.

A tabular Q learning agent is trained in two settings. One setting uses a clean reward signal that reflects actual task success. The other setting introduces stochastic reward corruption that simulates broken sensors, proxy objectives, or implementation flaws. The agent optimizes against the corrupted signal and learns behavior that diverges from the designer’s intent.

Training logs track both the reward observed by the agent and the true reward produced by the environment. Plots generated at the end of training visualize the gap between perceived success and real performance, highlighting how reward hacking can emerge even in simple systems.

The code is intentionally minimal and self contained. All logic is implemented in a single Python file and uses standard libraries. The experiment is deterministic under fixed random seeds and is suitable for demonstration, education, or interview discussion around alignment failures and safety in reinforcement learning.

## Requirements

Python 3.9 or newer is recommended. The code depends only on numpy, matplotlib, and tqdm.

## Running the Experiment

Execute the script directly using Python. Two agents will be trained sequentially. One agent operates under a clean reward signal. The second agent is trained with access to a corrupted reward channel. A plot comparing true reward and perceived reward will be displayed after training completes.

## Purpose

The purpose of the repository is to provide a concrete and reproducible example of reward hacking. The environment demonstrates how optimizing an imperfect objective can lead to coherent but unintended behavior. The experiment is small by design but captures failure modes that scale to more complex real world systems.

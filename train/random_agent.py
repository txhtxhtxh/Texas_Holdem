import numpy as np
import random
from br import BRAgent


class RandomAgent:
    def __init__(self, game):
        self.legal_action_tree = np.load(f"../data/{game}_legal_action_tree.npy", allow_pickle=True).item()

    def step(self, obs):
        path = obs['path']
        _, _, action_set = self.legal_action_tree[path]
        action_index = random.choice(range(len(action_set)))
        return action_index

class FoldAgent:
    def __init__(self, game):
        self.legal_action_tree = np.load(f"../data/{game}_legal_action_tree.npy", allow_pickle=True).item()
        self.policy = {}

    def get_policy(self):
        for history in self.legal_action_tree.keys():
            _, _, legal_actions = self.legal_action_tree[history]
            self.policy[history] = [0.] * len(legal_actions)
            self.policy[history][0] = 1.
        np.save("../data/fold_agent", self.policy)

    def step(self, obs):
        return 0

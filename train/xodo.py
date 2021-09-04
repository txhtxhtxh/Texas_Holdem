import random
import numpy as np
from copy import deepcopy
from br import BRAgent
from cfr import CFRAgent


class OXDOAgent:
    def __init__(self, game, encode_cards, fold_encode, deck, get_winner, round_card_num, deal_pokers):
        self.nash_finder = CFRAgent(game, encode_cards, fold_encode, deck, get_winner, round_card_num, deal_pokers)
        self.br = BRAgent(game, encode_cards, fold_encode, deck, get_winner, round_card_num, deal_pokers)
        self.game = game
        self.action_tree = {}
        self.player_num = 2
        self.discount = 0.9
        self.iterations = 0
        self.infoset_counts = 0
        self.from_scratch = True
        self.all_cards = deal_pokers(set(deck))

        # regret and policy for the agent
        self.regret = {}
        self.average_policy = {}

        self.legal_action_tree = np.load(f"../data/{game}_legal_action_tree.npy", allow_pickle=True).item()
        self.game_result_tree = np.load(f"../data/{game}_game_result_tree.npy", allow_pickle=True).item()
        # Init restricted_game
        self.restricted_game = {}

        # Init utils
        self.encode_cards, self.fold_encode, self.get_winner, self.round_card_num = \
            encode_cards, fold_encode, get_winner, round_card_num

    def step(self, obs):
        card = obs['card']
        path = obs['path']

        if card in self.average_policy:
            if path in self.average_policy[card]:
                action_p = self.average_policy[card][path]
                action_index = np.argmax(action_p)
                return action_index

        _, _, action_set = self.legal_action_tree[path]
        action_index = random.choice(range(len(action_set)))
        return action_index

    def add_strategy(self, key, history, br_action):
        if key in self.restricted_game:
            if history in self.restricted_game[key]:
                if br_action not in self.restricted_game[key][history]:
                    self.restricted_game[key][history].append(br_action)
                else:
                    return
            else:
                self.restricted_game[key][history] = [br_action]
        else:
            self.restricted_game[key] = {history: [br_action]}
        # If there are new actions in the br_list, train from scratch
        self.from_scratch = True

    def extend_policy(self, policy):
        full_policy = deepcopy(policy)
        for c, v in policy.items():
            for hist, value in v.items():
                _, _, legal_action = self.legal_action_tree[hist]
                restricted_actions = self.restricted_game[c][hist]
                full_policy[c][hist] = np.zeros(len(legal_action))
                for i, action in enumerate(restricted_actions):
                    full_policy[c][hist][legal_action.index(action)] = policy[c][hist][i]
        return full_policy

    def train(self):
        # Getting epsilon Nash
        nash_policy = self.nash_finder.train(
            restricted_tree=self.restricted_game,
            from_scratch=self.from_scratch)
        self.from_scratch = False

        full_nash_policy = self.extend_policy(nash_policy)
        exp, br_policy = self.br.exploitability(full_nash_policy, br=True)

        for key, policy in br_policy.items():
            for history, action_prob in policy.items():
                _, _, legal_actions = self.legal_action_tree[history]
                br_action = legal_actions[np.argmax(action_prob)]
                self.add_strategy(key, history, br_action)

        self.infoset_counts = self.nash_finder.infoset_counts + self.br.infoset_counts
        return full_nash_policy

    def get_payoff(self, cards, cur_node, player_id):
        result = self.get_winner(cards)
        in_chips_list, win_cal = self.game_result_tree[cur_node]
        if cur_node and cur_node[-1] != self.fold_encode:  # allin or river_call
            payoff = result[player_id] * sum(in_chips_list) - in_chips_list[player_id]
        else:
            payoff = win_cal[player_id]
        return payoff

    def save_agent(self, output_path):
        np.save(output_path, [self.regret, self.average_policy])

    def load_agent(self, data_pth):
        self.regret, self.average_policy = np.load(data_pth, allow_pickle=True)

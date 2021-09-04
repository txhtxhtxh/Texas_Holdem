import random
import numpy as np


class MCCFRAgent:
    def __init__(self, game, encode_cards, fold_encode, deck, get_winner, round_card_num, deal_pokers):
        self.legal_action_tree = np.load(f"../data/{game}_legal_action_tree.npy", allow_pickle=True).item()
        self.game_result_tree = np.load(f"../data/{game}_game_result_tree.npy", allow_pickle=True).item()
        self.restricted_tree = {}
        self.action_tree = {}
        self.player_num = 2
        self.result = None
        self.discount = 0.9
        self.iterations = 0
        self.infoset_counts = 0
        self.all_cards = deal_pokers(set(deck))

        # regret and policy for the agent
        self.regret = {}
        self.average_policy = {}

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

    def train(self, restricted_tree=None, from_scratch=False):
        self.iterations += 1
        self.restricted_tree = restricted_tree
        if from_scratch:
            self.average_policy = {}
            self.regret = {}

        # Training
        for cards in self.all_cards:
            for player_id in range(self.player_num):
                self.traverse('', player_id, cards)

        return self.average_policy

    def traverse(self, cur_node, real_id, cards):
        if cur_node in self.game_result_tree:
            payoff = self.get_payoff(cards, cur_node, real_id)
            return payoff

        cur_id, cur_round, legal_action = self.legal_action_tree[cur_node]
        cards_encoded = self.encode_cards(cards, cur_id, cur_round)
        restricted_legal_action = self.get_restricted_actions(cards_encoded, cur_node, legal_action)
        action_num = len(restricted_legal_action)
        action_prob = self.regret_matching(cards_encoded, cur_node, action_num)

        if cur_id != real_id:
            action = random.choices(legal_action, action_prob)[0]
            value = self.traverse(cur_node + action, real_id, cards)
        else:
            action_values = [self.traverse(cur_node + action, real_id, cards) for action in restricted_legal_action]
            value = np.dot(action_prob, action_values)

            # Update
            self.regret[cards_encoded][cur_node] += action_values - value
            self.average_policy[cards_encoded][cur_node] += action_prob * self.iterations

        return value

    def get_restricted_actions(self, cards_encoded, cur_node, legal_action):
        if self.restricted_tree == None:
            return legal_action
        if cards_encoded in self.restricted_tree:
            if cur_node in self.restricted_tree[cards_encoded]:
                return self.restricted_tree[cards_encoded][cur_node]
        return []

    def regret_matching(self, cards_encoded, history, action_num):
        regrets = self.get_regret(cards_encoded, history, action_num)
        pos_regrets = np.array([max(i, 0) for i in regrets])
        if sum(pos_regrets) > 0:
            return pos_regrets / sum(pos_regrets)
        return np.ones(action_num) / action_num

    def get_regret(self, cards_encoded, history, action_num):
        if cards_encoded in self.regret:
            if history in self.regret[cards_encoded]:
                if action_num > len(self.regret[cards_encoded][history]):
                    self.regret[cards_encoded][history] = np.append(self.regret[cards_encoded][history], 0.)
                return self.regret[cards_encoded][history]
            self.regret[cards_encoded][history] = np.zeros(action_num)
            return self.regret[cards_encoded][history]
        self.regret[cards_encoded] = {history: np.zeros(action_num)}
        self.average_policy[cards_encoded] = {}
        return self.regret[cards_encoded][history]

    def update_policy(self, cards_encoded, cur_node, policy):
        if cards_encoded in self.average_policy:
            if cur_node in self.average_policy[cards_encoded]:
                if len(policy) > len(self.average_policy[cards_encoded][cur_node]):
                    self.average_policy[cards_encoded][cur_node] = \
                        np.append(self.average_policy[cards_encoded][cur_node], 0.)
                self.average_policy[cards_encoded][cur_node] += policy
            else:
                self.average_policy[cards_encoded][cur_node] = policy

    def get_payoff(self, cards, cur_node, player_id):
        result = self.get_winner(cards)
        in_chips_list, win_cal = self.game_result_tree[cur_node]
        if cur_node and cur_node[-1] != self.fold_encode:  # allin or river_call
            payoff = result[player_id] * sum(in_chips_list) - in_chips_list[player_id]
        else:
            payoff = win_cal[player_id]
        return payoff

    def card2str(self, cards, cur_id, cur_round):
        return self.encode_cards(cards[cur_id][:self.round_card_num[cur_round]])

    def save_agent(self, output_path):
        np.save(output_path, [self.regret, self.average_policy])

    def load_agent(self, data_pth):
        self.regret, self.average_policy = np.load(data_pth, allow_pickle=True)

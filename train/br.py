import numpy as np
from copy import deepcopy
from utils_leduc import encode_cards, fold_encode, deck, get_winner, round_card_num, deal_pokers


class BRAgent:
    def __init__(self, game):
        self.legal_action_tree = np.load(f"../data/{game}_legal_action_tree.npy", allow_pickle=True).item()
        self.game_result_tree = np.load(f"../data/{game}_game_result_tree.npy", allow_pickle=True).item()
        self.action_tree = {}
        self.player_num = 2
        if game == 'limit':
            self.round_num = 4
        else:
            self.round_num = 2
        self.result = None
        self.game = game

        # tested policy and BR policy
        self.tested_policy = {}
        self.best_response = {}
        self.all_cards = deal_pokers(set(deck))

    def get_payoff(self, cards, cur_node, player_id):
        result = get_winner(cards)
        in_chips_list, win_cal = self.game_result_tree[cur_node]
        if cur_node and cur_node[-1] != fold_encode:  # allin or river_call
            payoff = result[player_id] * sum(in_chips_list) - in_chips_list[player_id]
        else:
            payoff = win_cal[player_id]
        return payoff

    def encode_cards(self, cards, cur_id, cur_round):
        return encode_cards(cards[cur_id][:round_card_num[cur_round]])

    def calc_ev(self, cur_node, cards, br_id):
        if cur_node in self.game_result_tree:
            return self.get_payoff(cards, cur_node, br_id)

        cur_id, cur_round, legal_action = self.legal_action_tree[cur_node]
        my_card = self.encode_cards(cards, cur_id, cur_round)
        if cur_id == br_id:
            action_prob = self.best_response[my_card][cur_node]
        else:
            action_prob = self.tested_policy[my_card][cur_node]
        action_prob = action_prob / sum(action_prob)
        next_evs = [self.calc_ev(cur_node + a, cards, br_id) for a in legal_action]
        return np.dot(action_prob, next_evs)

    def get_best_response(self, cur_node, cards, chance_prob, br_id):
        if cur_node in self.game_result_tree:
            return self.get_payoff(cards, cur_node, br_id)

        cur_id, cur_round, legal_action = self.legal_action_tree[cur_node]
        cards_encoded = self.encode_cards(cards, cur_id, cur_round)
        action_prob = self.tested_policy[cards_encoded][cur_node]
        action_prob = action_prob / sum(action_prob)
        action_values = np.zeros(len(action_prob))
        for i, action in enumerate(legal_action):
            if cur_id != br_id:
                value = self.get_best_response(cur_node + action, cards, action_prob[i] * chance_prob, br_id)
            else:
                value = self.get_best_response(cur_node + action, cards, chance_prob, br_id)
            action_values[i] = value

        if cur_id == br_id:
            self.best_response[cards_encoded][cur_node] += action_values * chance_prob
            return max(action_values)
        else:
            return np.dot(action_prob, action_values)

    def exploitability(self):
        result_list = []
        for cards in self.all_cards:
            for br_id in range(self.player_num):
                self.get_best_response('', cards, 1., br_id)

        for k, v in self.best_response.items():
            for history, values in v.items():
                self.best_response[k][history] = np.where(values == np.max(values), 1, 0)

        for cards in self.all_cards:
            for br_id in range(self.player_num):
                result_list.append(self.calc_ev("", cards, br_id))

        return sum(result_list) / len(result_list)

    def load_policy(self, data_pth):
        _, self.tested_policy = np.load(data_pth, allow_pickle=True)
        self.best_response = deepcopy(self.tested_policy)

        for k, v in self.best_response.items():
            for h, j in v.items():
                self.best_response[k][h] = np.zeros(len(j))

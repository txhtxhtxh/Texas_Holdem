import numpy as np
from copy import deepcopy


class BRAgent:
    def __init__(self, game, encode_cards, fold_encode, deck, get_winner, round_card_num, deal_pokers):
        self.legal_action_tree = np.load(f"../data/{game}_legal_action_tree.npy", allow_pickle=True).item()
        self.game_result_tree = np.load(f"../data/{game}_game_result_tree.npy", allow_pickle=True).item()
        self.action_tree = {}
        self.player_num = 2
        if game == 'limit':
            self.round_num = 4
        else:
            self.round_num = 2
        self.infoset_counts = 0
        self.game = game

        # tested policy and BR policy
        self.tested_policy = {}
        self.best_response = {}
        self.restricted_tree = {}
        self.all_cards = deal_pokers(set(deck))

        # Init utils
        self.encode_cards, self.fold_encode, self.get_winner, self.round_card_num = \
            encode_cards, fold_encode, get_winner, round_card_num

    def exploitability(self, tested_policy, restricted_tree=None, br=False):
        self.restricted_tree = restricted_tree
        self.get_best_response(tested_policy)
        result_list = []

        for cards in self.all_cards:
            for br_id in range(self.player_num):
                result_list.append(self.calc_ev("", cards, br_id))

        if br:
            return sum(result_list) / len(result_list), self.best_response
        else:
            return sum(result_list) / len(result_list)

    def get_best_response(self, tested_policy):
        self.tested_policy = deepcopy(tested_policy)

        self.init()

        for br_id in range(self.player_num):
            for cards in self.all_cards:
                self.br_traverse('', cards, 1., br_id)

        for k, v in self.best_response.items():
            for history, values in v.items():
                action_prob = np.where(values == np.max(values), 1, 0)
                self.best_response[k][history] = action_prob / np.sum(action_prob)

        return self.best_response

    def init(self):
        '''
        1.Fill in the blank action probs with uniform distribution
        2.Initialize BR with zeros value everywhere
        '''
        self.best_response = {}
        for cards in self.all_cards:
            for history, value in self.legal_action_tree.items():
                cur_id, cur_round, ori_legal_action = value
                cards_encode = self.card2str(cards, cur_id, cur_round)
                action_num = len(self.get_restricted_actions(cards_encode, history, ori_legal_action))

                if cards_encode in self.best_response:
                    if history not in self.best_response[cards_encode]:
                        self.best_response[cards_encode][history] = np.zeros(action_num)
                else:
                    self.best_response[cards_encode] = {history: np.zeros(action_num)}

                if cards_encode in self.tested_policy:
                    if history in self.tested_policy[cards_encode]:
                        if sum(self.tested_policy[cards_encode][history]) > 0:
                            self.tested_policy[cards_encode][history] /= sum(self.tested_policy[cards_encode][history])
                        else:
                            self.tested_policy[cards_encode][history] = np.ones(action_num) / action_num
                    else:
                        self.tested_policy[cards_encode][history] = np.ones(action_num) / action_num
                else:
                    self.tested_policy[cards_encode] = {history: np.ones(action_num) / action_num}

    def br_traverse(self, cur_node, cards, chance_prob, br_id):
        self.infoset_counts += 1

        if cur_node in self.game_result_tree:
            return self.get_payoff(cards, cur_node, br_id)

        cur_id, cur_round, legal_action = self.legal_action_tree[cur_node]
        cards_encoded = self.card2str(cards, cur_id, cur_round)
        restricted_actions = self.get_restricted_actions(cards_encoded, cur_node, legal_action)
        action_prob = self.tested_policy[cards_encoded][cur_node]
        action_values = np.zeros(len(action_prob))

        for i, action in enumerate(restricted_actions):
            if cur_id == br_id:
                value = self.br_traverse(cur_node + action, cards, chance_prob, br_id)
            else:
                value = self.br_traverse(cur_node + action, cards, action_prob[i] * chance_prob, br_id)
            action_values[i] = value

        if cur_id == br_id:
            self.best_response[cards_encoded][cur_node] += action_values * chance_prob
            return max(action_values)
        else:
            return np.dot(action_prob, action_values)

    def calc_ev(self, cur_node, cards, br_id):
        if cur_node in self.game_result_tree:
            return self.get_payoff(cards, cur_node, br_id)

        cur_id, cur_round, legal_action = self.legal_action_tree[cur_node]
        cards_encoded = self.card2str(cards, cur_id, cur_round)
        restricted_actions = self.get_restricted_actions(cards_encoded, cur_node, legal_action)

        if cur_id == br_id:
            action_prob = self.best_response[cards_encoded][cur_node]
        else:
            action_prob = self.tested_policy[cards_encoded][cur_node]

        next_evs = [self.calc_ev(cur_node + a, cards, br_id) for a in restricted_actions]

        return np.dot(action_prob, next_evs)

    def load_policy(self, data_pth):
        _, self.tested_policy = np.load(data_pth, allow_pickle=True)

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

    def get_restricted_actions(self, cards_encoded, cur_node, legal_action):
        if self.restricted_tree == None:
            return legal_action
        if cards_encoded in self.restricted_tree:
            if cur_node in self.restricted_tree[cards_encoded]:
                return self.restricted_tree[cards_encoded][cur_node]
        return []

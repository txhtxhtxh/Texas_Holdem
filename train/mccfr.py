import random
import numpy as np
from test import multi_test
from br import BRAgent
from random_agent import RandomAgent
from utils_leduc import encode_cards, fold_encode, deck, get_winner, round_card_num, deal_pokers


class MCCFRAgent:
    def __init__(self, game):
        self.legal_action_tree = np.load(f"../data/{game}_legal_action_tree.npy", allow_pickle=True).item()
        self.game_result_tree = np.load(f"../data/{game}_game_result_tree.npy", allow_pickle=True).item()
        self.game = game
        self.action_tree = {}
        self.player_num = 2
        if game == 'limit':
            self.round_num = 4
        else:
            self.round_num = 2
        self.result = None
        self.discount = 0.9
        self.iterations = 0
        self.all_cards = deal_pokers(set(deck))

        # regret and policy for the agent
        self.regret = {}
        self.average_policy = {}

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

    def train(self):
        self.iterations += 1

        # Training
        cards = random.choice(self.all_cards)
        for player_id in range(self.player_num):
            self.traverse('', player_id, cards)

    def traverse(self, cur_node, real_id, cards):
        if cur_node in self.game_result_tree:
            payoff = self.get_payoff(cards, cur_node, real_id)
            return payoff

        cur_id, cur_round, legal_action = self.legal_action_tree[cur_node]
        cards_encoded = self.encode_cards(cards, cur_id, cur_round)
        action_prob = self.regret_matching(cards_encoded, cur_node)

        if cur_id != real_id:
            action = random.choices(legal_action, action_prob)[0]
            value = self.traverse(cur_node + action, real_id, cards)
        else:
            action_values = [self.traverse(cur_node + action, real_id, cards) for action in legal_action]
            value = np.dot(action_prob, action_values)

            # Update
            self.regret[cards_encoded][cur_node] += action_values - value
            self.average_policy[cards_encoded][cur_node] += action_prob * self.iterations
        return value

    def regret_matching(self, cards_encoded, history):
        regrets, _ = self.get_regret(cards_encoded, history)
        pos_regrets = np.array([max(i, 0) for i in regrets])
        if sum(pos_regrets) > 0:
            return pos_regrets / sum(pos_regrets)
        return np.ones(len(pos_regrets)) / len(pos_regrets)

    def get_regret(self, cards_encoded, history):
        action_num = len(self.legal_action_tree[history][2])
        if cards_encoded in self.regret:
            if history in self.regret[cards_encoded]:
                return self.regret[cards_encoded][history], self.average_policy[cards_encoded][history]
            self.regret[cards_encoded][history] = np.zeros(action_num)
            self.average_policy[cards_encoded][history] = np.ones(action_num) / action_num
            return self.regret[cards_encoded][history], self.average_policy[cards_encoded][history]
        self.regret[cards_encoded] = {history: np.zeros(action_num)}
        self.average_policy[cards_encoded] = {history: np.ones(action_num) / action_num}
        return self.regret[cards_encoded][history], self.average_policy[cards_encoded][history]

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

    def save_agent(self, output_path):
        np.save(output_path, [self.regret, self.average_policy])

    def load_agent(self, data_pth):
        self.regret, self.average_policy = np.load(data_pth, allow_pickle=True)


process_num = 20
testing_num = 1000
training_times = 10000
test_adjunct = training_times
game = "leduc"
mccfr_agent = MCCFRAgent(game)
random_agent = RandomAgent(game)
br_agent = BRAgent(game)


print("start training")
results = []
for train_step in range(training_times):
    mccfr_agent.train()
    if (train_step + 1) % test_adjunct == 0:
        output_path = f"../data/cfr_train_data_{train_step}.npy"
        mccfr_agent.save_agent(output_path)
        # multi_test([cfr_agent, random_agent], process_num=process_num, testing_num=testing_num, game=cfr_agent.game)
        exp = br_agent.exploitability(output_path)
        print(train_step, exp)
        results.append(exp)
print("Done")
print(results)

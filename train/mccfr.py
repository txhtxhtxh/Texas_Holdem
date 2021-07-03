import random
import numpy as np
from test import multi_test
from random_agent import RandomAgent
from br import BRAgent
from utils_leduc import random_deal, encode_cards, fold_encode, deck, get_winner

class MCCFRAgent:
    def __init__(self, game):
        self.legal_action_tree = np.load(f"../data/{game}_legal_action_tree.npy", allow_pickle=True).item()
        self.game_result_tree = np.load(f"../data/{game}_game_result_tree.npy", allow_pickle=True).item()
        self.game = game
        self.encoded_cards = []
        self.action_tree = {}
        self.player_num = 2
        self.round_num = 4
        self.result = None
        self.discount = 0.9

        # regret and policy for each traverse
        self.reg_tree = [[dict() for _ in range(self.round_num)] for _ in range(2)]
        self.policy_tree = [[dict() for _ in range(self.round_num)] for _ in range(2)]

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

    def train(self, cards):
        # get result based on cards
        self.result = get_winner(cards)

        # Encode cards for four rounds
        player_poker = cards
        self.encoded_cards = []
        for player_id in range(self.player_num):
            temp_cards = []
            for i in range(self.round_num):

                temp_cards.append(encode_cards(player_poker[player_id][:(i + 1)]))
            self.encoded_cards.append(temp_cards)

        self.traverse('', 0)
        self.traverse('', 1)
        # Update
        self.update()

    def traverse(self, cur_node, real_id):
        if cur_node in self.game_result_tree:
            in_chips_list, win_cal = self.game_result_tree[cur_node]
            if cur_node and cur_node[-1] != fold_encode:  # allin or river_call
                payoff = self.result[real_id] * sum(in_chips_list) - in_chips_list[real_id]
            else:
                payoff = win_cal[real_id]
            return payoff
        cur_id, cur_round, legal_action = self.legal_action_tree[cur_node]
        action_num = len(legal_action)
        action_prob = self.regret_matching(self.encoded_cards[cur_id][cur_round], cur_node)

        if cur_id == real_id:
            value = 0.
            action_value_list = []
            for i in range(len(legal_action)):
                action, action_p = legal_action[i], action_prob[i]
                action_value = self.traverse(cur_node + action, real_id)
                action_value_list.append(action_value)
                value += action_p * action_value

            # get regret and policy
            self.reg_tree[cur_id][cur_round][cur_node] = [action_value_list[i] - value for i in range(action_num)]
            self.policy_tree[cur_id][cur_round][cur_node] = action_prob
        else:
            action = random.choices(legal_action, action_prob)[0]
            value = self.traverse(cur_node + action, real_id)

        return value

    def regret_matching(self, cards_encoded, history):
        regret, _ = self.get_regret(cards_encoded, history)
        regret = [max(i, 0) for i in regret]
        if sum(regret) > 0:
            return [i / sum(regret) for i in regret]
        return [1. / len(regret)] * len(regret)

    def get_regret(self, cards_encoded, history):
        action_num = len(self.legal_action_tree[history][2])
        if cards_encoded in self.regret:
            if history in self.regret[cards_encoded]:
                return self.regret[cards_encoded][history], self.average_policy[cards_encoded][history]
            self.regret[cards_encoded][history] = [0.] * action_num
            self.average_policy[cards_encoded][history] = [1. / action_num] * action_num
            return self.regret[cards_encoded][history], self.average_policy[cards_encoded][history]
        self.regret[cards_encoded] = {history: [0.] * action_num}
        self.average_policy[cards_encoded] = {history: [1. / action_num] * action_num}
        return self.regret[cards_encoded][history], self.average_policy[cards_encoded][history]

    def update(self):
        for cur_id in range(2):
            for cur_round in range(self.round_num):
                cards_encoded = self.encoded_cards[cur_id][cur_round]
                for history, cur_reg in self.reg_tree[cur_id][cur_round].items():
                    last_reg, last_avp = self.get_regret(cards_encoded, history)

                    # Cumulate regret
                    self.regret[cards_encoded][history] = \
                        [cur_reg[i] for i in range(len(last_reg))]
                        #[last_reg[i] * self.discount + cur_reg[i] for i in range(len(last_reg))]

                    # Cumulate policy
                    cur_avp = self.policy_tree[cur_id][cur_round][history]
                    self.average_policy[cards_encoded][history] = \
                        [last_avp[i] + cur_avp[i] for i in range(len(last_avp))]

    def save_agent(self, output_path):
        np.save(output_path, [self.regret, self.average_policy])

    def load_agent(self, data_pth):
        self.regret, self.average_policy = np.load(data_pth, allow_pickle=True)


training_times = 50000
test_adjunct = training_times

process_num = 20
testing_num = 1000
cfr_agent = MCCFRAgent("leduc")
random_agent = RandomAgent("leduc")
br_agent = BRAgent('leduc')

print("start training")
res_poker = set(deck)
for train_step in range(training_times):
    cfr_agent.train(random_deal(res_poker))
    if (train_step + 1) % test_adjunct == 0:
        output_path = f"../data/cfr_train_data_{train_step}.npy"
        cfr_agent.save_agent(output_path)
        #multi_test([cfr_agent, random_agent], process_num=process_num, testing_num=testing_num, game=cfr_agent.game)
        br_agent.load_policy(output_path)
        print(br_agent.exploitability())
print("Done")

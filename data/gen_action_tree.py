import numpy as np
from copy import deepcopy
from env.env import Holdem_hand_env
from _collections import defaultdict
from train.utils_limit import action_encode, fold_encode


class LimitTexasTree:
    '''
    This class is to generate a action tree for limit texas Hold'em poker
    '''
    def __init__(self):
        self.class_name = "limit"
        self.player_num = 2
        self.legal_tree = defaultdict(list)
        self.end_tree = {}

    def legal_action_space(self, round_counter, raise_times):
        if round_counter <= 1:
            if raise_times <= 2:
                return ['f', 'c', 'r']
            else:
                return ['f', 'c']
        else:
            if raise_times <= 3:
                return ['f', 'c', 'r']
            else:
                return ['f', 'c']

    def cal_action(self, str_action, observation, bb, round_counter):
        if str_action == 'f':
            return -1

        call_amount = max(observation[1]['res_chip']) - min(observation[1]['res_chip'])
        if str_action == 'c':
            return call_amount

        if str_action == 'a':
            return observation[1]['res_chip'][observation[0]]

        if round_counter <= 1:
            return bb + call_amount
        else:
            return bb * 2 + call_amount

    def next_node(self, env, action_sequence, observation, in_chips_list, last_round, raise_times):
        if observation[3]:
            win_cal = None
            if action_sequence and action_sequence[-1] == fold_encode:
                win_cal = observation[2]
            self.end_tree[action_sequence] = [[min(min(in_chips_list),i) for i in in_chips_list],win_cal]
            return
        action_set = []
        bb = env.bb
        cur_env = deepcopy(env)
        cur_player = cur_env.cur_player_index
        round_counter = observation[1]['round']
        new_raise_times = raise_times

        if round_counter > last_round:
            new_raise_times = 0
        for str_action in self.legal_action_space(round_counter, raise_times):
            if str_action == 'r':
                new_raise_times = raise_times + 1
            encode_action = action_encode[str_action]
            action = self.cal_action(str_action, observation, bb, new_raise_times)
            self.legal_tree[action_sequence].append(encode_action)
            if (env.check_action(action) > -2) and (action not in action_set):
                action_set.append(action)
                new_observation = cur_env.step(action)
                action2 = max(action, 0)
                next_in_chips_list = [in_chips_list[i]+action2
                                      if i == cur_player else in_chips_list[i] for i in range(2)]
                self.next_node(cur_env, action_sequence + encode_action,
                               new_observation, next_in_chips_list, observation[1]['round'], new_raise_times)
                del cur_env
                cur_env = deepcopy(env)
        if action_sequence in self.legal_tree:
            self.legal_tree[action_sequence] = [cur_player, round_counter, self.legal_tree[action_sequence]]

    def generate_tree(self, chip_list, sb, bb, ante):
        env = Holdem_hand_env(
            agent_list=[i for i in range(2)],
            chips_list=chip_list,
            sb=sb,
            bb=bb,
            D=0,
            straddle=0,
            ante=ante)
        # observation consists of: cur_player_index,  obs,  R,  done, info
        observation = env.reset()
        # Start building a tree
        self.next_node(env, '', observation, [sb, bb], 0, 0)
        np.save(f"{self.class_name}_legal_action_tree.npy", self.legal_tree, allow_pickle=True)
        np.save(f"{self.class_name}_game_result_tree.npy", self.end_tree, allow_pickle=True)


class LeducHoldem:
    '''
    This class is to generate a action tree for Leduc texas Hold'em poker
    '''
    def __init__(self):
        self.player_num = 2
        self.end_tree = {}
        self.class_name = "leduc"
        self.legal_tree = defaultdict(list)

    def legal_action_space(self, raise_times):
        if raise_times < 2:
            return ['f', 'c', 'r']
        else:
            return ['f', 'c']

    def cal_action(self, str_action, observation, round_counter):
        if str_action == 'f':
            return -1

        call_amount = max(observation[1]['res_chip']) - min(observation[1]['res_chip'])
        if str_action == 'c':
            return call_amount

        if round_counter <= 1:
            return 2 + call_amount
        else:
            return 4 + call_amount

    def next_node(self, env, action_sequence, observation, in_chips_list, last_round, raise_times):
        round_counter = observation[1]['round']
        if observation[3] or round_counter == 3:
            win_cal = None
            if action_sequence and action_sequence[-1] == fold_encode:
                win_cal = observation[2]
            self.end_tree[action_sequence] = [[min(min(in_chips_list),i) for i in in_chips_list], win_cal]
            return
        action_set = []
        cur_env = deepcopy(env)
        cur_player = cur_env.cur_player_index
        new_raise_times = raise_times

        if round_counter > last_round:
            new_raise_times = 0

        for str_action in self.legal_action_space(raise_times):
            if str_action == 'r':
                new_raise_times = raise_times + 1
            encode_action = action_encode[str_action]
            action = self.cal_action(str_action, observation, round_counter)
            self.legal_tree[action_sequence].append(encode_action)
            if (env.check_action(action) > -2) and (action not in action_set):
                action_set.append(action)
                new_observation = cur_env.step(action)
                action2 = max(action, 0)
                next_in_chips_list = [in_chips_list[i]+action2
                                      if i == cur_player else in_chips_list[i] for i in range(2)]
                self.next_node(cur_env, action_sequence + encode_action,
                               new_observation, next_in_chips_list, observation[1]['round'], new_raise_times)
                del cur_env
                cur_env = deepcopy(env)
        if action_sequence in self.legal_tree:
            self.legal_tree[action_sequence] = [cur_player, round_counter - 1, self.legal_tree[action_sequence]]

    def generate_tree(self, chip_list, sb, bb, ante):
        env = Holdem_hand_env(
            agent_list=[i for i in range(2)],
            chips_list=chip_list,
            sb=sb,
            bb=bb,
            D=0,
            straddle=0,
            ante=ante)
        # observation consists of: cur_player_index,  obs,  R,  done, info
        env.reset()
        observation = env.step(0)
        # Start building a tree
        # starting round is set to be one
        self.next_node(env=env,
                       action_sequence='',
                       observation=observation,
                       in_chips_list=[sb + ante, bb + ante],
                       last_round=1,
                       raise_times=0)
        np.save(f"{self.class_name}_legal_action_tree.npy", self.legal_tree, allow_pickle=True)
        np.save(f"{self.class_name}_game_result_tree.npy", self.end_tree, allow_pickle=True)


# # Limit Texas
# tree = LimitTexasTree()
# tree.generate_tree(chip_list=[200] * 2,
#                    sb=1,
#                    bb=2,
#                    ante=0)
#

# Leduc
# tree = LeducHoldem()
# tree.generate_tree(chip_list=[199] * 2,
#                    sb=0,
#                    bb=0,
#                    ante=1.)


led = np.load("../data/leduc_legal_action_tree.npy").item()
led_result = np.load("../data/leduc_game_result_tree.npy").item()

print(led)
print(led_result)

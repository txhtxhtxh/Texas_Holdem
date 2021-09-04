import pickle
import random_agent
import time

import numpy as np
from treys import Evaluator
from treys import Card

evaluator = Evaluator()
from utils_limit import encode_cards, random_deal, fold_encode, deck, random_pick, round_card_num
from multiprocessing import Pool


def one_hand_game(game, player_poker, table_poker, agents, real_id):
    legal_action_tree = np.load(f"../data/{game}_legal_action_tree.npy", allow_pickle=True).item()
    game_result_tree = np.load(f"../data/{game}_game_result_tree.npy", allow_pickle=True).item()
    path = ''
    while 1:
        # If terminal:
        if path in game_result_tree:
            in_chips_list, win_cal = game_result_tree[path]
            if path and path[-1] != fold_encode:  # allin or  River_call
                A_score = - evaluator._seven([Card.new(c) for c in player_poker[0] + table_poker])
                B_score = - evaluator._seven([Card.new(c) for c in player_poker[1] + table_poker])
                if A_score == B_score:
                    allin_win_pa = 0.5
                else:
                    allin_win_pa = 1. if A_score > B_score else 0
                get_score = allin_win_pa * sum(in_chips_list)
                sb_win_score = get_score - in_chips_list[real_id]
            else:
                sb_win_score = win_cal[real_id]
            return sb_win_score

        cur_id, cur_round, legal_action_list = legal_action_tree[path]
        cards_encoded = encode_cards(player_poker[real_id] + table_poker[:(round_card_num[cur_round] - 2)])
        action_index = agents[cur_id].step(
            {'path': path, 'card': cards_encoded}
        )
        action = legal_action_list[action_index]
        path += action


def test(input):
    battle_agents, bl_time, game = input
    score_list = []
    res_poker = set(deck)
    for i in range(bl_time):
        player_poker, table_poker = random_deal(res_poker)
        score_list.append(one_hand_game(game, player_poker, table_poker, battle_agents, 0))
        battle_agents = [battle_agents[1], battle_agents[0]]
        score_list.append(one_hand_game(game, player_poker, table_poker, battle_agents, 1))
    result = sum(score_list) / len(score_list)
    return result


def multi_test(agents, process_num, testing_num, game):
    working_pool = Pool(process_num)
    result_list = working_pool.map(test, [(agents, testing_num, game) for _ in range(process_num)])
    if game == 'leduc':
        print(sum(result_list) / len(result_list), "ante")
    else:
        print(1000 * sum(result_list) / len(result_list) / 2, "mbb")

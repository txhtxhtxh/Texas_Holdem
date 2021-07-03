import random
from itertools import combinations
from treys import Evaluator
from treys import Card


evaluator = Evaluator()
round_card_num = [2, 5, 6, 7]
round_num = len(round_card_num)
action_list = ['f', 'c', 'r', 'a']
action_encode_num = 'fcra'
action_encode = {action_list[i]: action_encode_num[i] for i in range(len(action_list))}
fold_encode = action_encode['f']
deck = ['2s', '2h', '2c', '2d', '3s', '3h', '3c', '4s', '4h']

# def encode_cards(card_list):
#
#     flu_c = 1
#     flu_map = dict()
#     encode_key = ''
#     for card in sorted_card_list:
#         num, flu = card
#         if flu in flu_map:
#             encode_key += num + flu_map[flu]
#         else:
#             encode_key += num + str(flu_c)
#             flu_map[flu] = str(flu_c)
#             flu_c += 1
#     return encode_key


def encode_cards(card_list):
    hands, public = sorted(card_list[:2]), sorted(card_list[2:])
    sorted_card_list = hands + public
    return "".join(sorted_card_list)


def get_winner(cards):
    player_poker, table_poker = cards
    # Getting the result of the final round: seven cards
    score_0 = - evaluator._seven([Card.new(c) for c in player_poker[0] + table_poker])
    score_1 = - evaluator._seven([Card.new(c) for c in player_poker[1] + table_poker])
    if score_0 == score_1:
        return [0.5, 0.5]
    else:
        result = [0., 0.]
        if score_0 > score_1:
            result[0] = 1.
        else:
            result[1] = 1.
        return result


def deal_pokers(res_poker):
    for com in combinations(res_poker, 2):
        com = list(com)
        res_poker2 = res_poker - set(com)
        for com2 in combinations(res_poker2, 2):
            com2 = list(com2)
            res_poker3 = res_poker2 - set(com2)
            for com3 in combinations(res_poker3, 3):
                res_poker4 = res_poker3 - set(com3)
                for com4 in combinations(res_poker4, 1):
                    res_poker5 = res_poker4 - set(com4)
                    for com5 in combinations(res_poker5, 1):
                        player_poker = [com, com2]
                        table_poker = list(com3) + list(com4) + list(com5)
                        yield [player_poker, table_poker]


def random_deal(cards):
    res_poker = list(cards)
    cards_index = random.sample(range(len(res_poker)), 9)
    hands = [[res_poker[cards_index[0]], res_poker[cards_index[1]]], [res_poker[cards_index[2]], res_poker[cards_index[3]]]]
    table = [res_poker[cards_index[4 + i]] for i in range(5)]
    return [hands, table]


def random_pick(pick_list, probs):
    num = random.random()
    sum_up = 0
    for i, p in enumerate(probs):
        if (num >= sum_up) and (num < sum_up + p):
            return pick_list[i]
        else:
            sum_up += p
    return pick_list[probs.index(max(probs))]
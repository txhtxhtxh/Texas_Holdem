import random
from itertools import combinations


round_card_num = [1]
round_num = len(round_card_num)
action_list = ['f', 'c', 'r']
action_encode_num = 'fcr'
action_encode = {action_list[i]: action_encode_num[i] for i in range(len(action_list))}
fold_encode = action_encode['f']
deck = ['2', '3', '4']

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


def deal_pokers(res_poker):
    result_list = []
    for com in combinations(res_poker, 1):
        com = list(com)
        res_poker2 = res_poker - set(com)
        for com2 in combinations(res_poker2, 1):
            com2 = list(com2)
            result_list.append([com,com2])
    return result_list


def encode_cards(card_list):
    return "".join(sorted(card_list))


def get_winner(cards):
    score = [eval(c[0]) for c in cards]
    result = [0., 0.]
    if score[0] > score[1]:
        result[0] = 1.
    else:
        result[1] = 1.
    return result



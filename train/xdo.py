# # CFR
import pickle
import copy
from train.utils_limit import to_key,round_card_num, mc_deal_pokers, F_encode,poker_9,all_round_num,all_round_card_list

# print(all_action_list)
# print(action_encode)
path = '../data/'
f1 = open(path + 'Legal_action_tree.npy','rb')
Legal_action_tree = pickle.load(f1)
f1.close()
# print(data[''])
# print(data.keys() )
f1 = open(path + 'End_game_tree.npy','rb')
End_game_tree = pickle.load(f1)
End_game_tree_key = set(End_game_tree.keys())
# print(End_game_tree)
f1.close()



# 动作树
# all_action_node = Legal_action_tree.keys()

round_action_tree = [dict()  for i in round_card_num]
for k ,v in Legal_action_tree.items():
    cur_id, cur_round, lega_action = v
    round_action_tree[cur_round][k] = v

round_card_action_tree = [ { node :{k:[1./len(v[2])]*len(v[2]) for k ,v in round_action_tree[cur_round].items()}
                             for node in all_round_card_list[cur_round] }
                           for cur_round in range(all_round_num)]

round_card_reg_tree = [ { node :{k:[0.]*len(v[2]) for k ,v in round_action_tree[cur_round].items()}
                             for node in all_round_card_list[cur_round] }
                           for cur_round in range(all_round_num)]

round_card_sump_tree = copy.deepcopy(round_card_reg_tree)

def get_action_tree(hand_card,table_card):
    # cards = hand_card + table_card
    all_round_key = [to_key(hand_card,table_card[:round_card_num[i]]) for i in range(4)]
    return [round_card_action_tree[i][all_round_key[i]] for i in range(4)],all_round_key



# 发牌
res_poker = set(poker_9)
player_poker = []

# print(time.time()-s)
from treys import Evaluator
from treys import Card
evaluator = Evaluator()

def get_reg_tree(player_poker, table_poker, action_p_tree_list,encode_node_list ):
    # reg_tree = [[dict() for j in range(all_round_num)] for i in range(2)]
    # sum_action_p = [[dict() for j in range(all_round_num)] for i in range(2)]
    A_score = - evaluator._seven([Card.new(c) for c in player_poker[0] + table_poker])
    B_score = - evaluator._seven([Card.new(c) for c in player_poker[1] + table_poker])
    if A_score == B_score:
        allin_win_pa = 0.5
    else:
        allin_win_pa =  1. if A_score > B_score else 0

    def get_value(cur_node,playerA_action_p,playerB_action_p ):
        end_msg = End_game_tree.get(cur_node)
        if end_msg:
            in_chips_list,win_cal =end_msg
            if cur_node and cur_node[-1] != F_encode:  # allin or  River_call
                get_score = allin_win_pa*sum(in_chips_list)
                sb_win_score =  get_score-in_chips_list[0]
            else:
                sb_win_score =  win_cal[0]
            return sb_win_score

        value = 0.
        cur_id,cur_round , lega_action  =Legal_action_tree[cur_node]
        action_p_list = action_p_tree_list[cur_id][cur_round][cur_node]
        action_num = len(action_p_list)
        action_value_list = [0.]*action_num
        for i in range(action_num):
            action = lega_action[i]
            action_p = action_p_list[i]
            if cur_id ==0:
                next_playerA_action_p  = action_p * playerA_action_p
                next_playerB_action_p  = playerB_action_p
            else:
                next_playerA_action_p =  playerA_action_p
                next_playerB_action_p = action_p * playerB_action_p
            action_sb_value = get_value(cur_node+action,next_playerA_action_p,next_playerB_action_p)
            action_value = -action_sb_value if cur_id==1 else action_sb_value
            action_value_list[i] = action_value
            # print(cur_node,action_p,action_value)
            value += action_p*action_value
        # action_reg_list = [action_value - value for action_value in action_value_list]
        # 累积遗憾值  # 累积概率
        action_reg_list = [0.]*action_num
        action_sum_p_list = [0.]*action_num
        if cur_id == 0:
            other_path_p = playerB_action_p
            # R_sign = 1
            my_path_p = playerA_action_p
        else:
            other_path_p = playerA_action_p
            # R_sign = -1
            my_path_p = playerB_action_p
        for i in range(action_num):
            action_reg_list[i] = other_path_p*(action_value_list[i]-value)#*R_sign
            action_sum_p_list[i] = action_p_list[i]*my_path_p
        cur_card_ec = encode_node_list[cur_id][cur_round]
        last_reg = round_card_reg_tree[cur_round][cur_card_ec][cur_node]
        sum_reg = [last_reg[i] + action_reg_list[i] for i in range(len(last_reg))]
        last_sump = round_card_sump_tree[cur_round][cur_card_ec][cur_node]
        round_card_sump_tree[cur_round][cur_card_ec][cur_node] = [last_sump[i] + action_sum_p_list[i] for i in range(len(last_sump))]

        sum_reg2 = [max(i, 0) for i in sum_reg]
        action_p = get_save_p(sum_reg2)
        round_card_action_tree[cur_round][cur_card_ec][cur_node] = action_p

        round_card_reg_tree[cur_round][cur_card_ec][cur_node] = sum_reg2   # CFR plus
        # round_card_reg_tree[cur_round][cur_card_ec][cur_node] = sum_reg    # CFR


        return value
    get_value('',1.,1.)
    # all_k = set()
    # for i in reg_tree:
    #     for j in i:
    #         for k in j.keys():
    #             all_k.add(k)
    # print(len(all_k),all_k)
    # print(sum([len(j.keys()) for i in reg_tree for j in i ]))
    # return [],[]#reg_tree,sum_action_p

def get_save_p(fz):
    fm = sum(fz)
    # print('fz :',h_index,root,sum(fz),fz)
    if fm > 0:
        return [i/fm for i in fz] # 更新手牌分布
    else:
        return  [1. / len(fz)] * len(fz)
# 更新
def update():
    # for cur_id, cur_round, cur_node, cur_reg, cur_sump ,cur_card_ec in reg_list:
    #     last_reg = round_card_reg_tree[cur_round][cur_card_ec][k]
    #     round_card_reg_tree[cur_round][cur_card_ec][k] = [last_reg[i] + cur_reg[i] for i in range(len(last_reg))]
    #     last_sump = round_card_sump_tree[cur_round][cur_card_ec][k]
    #     round_card_sump_tree[cur_round][cur_card_ec][k] = [last_sump[i] + cur_sump[i] for i in range(len(last_sump))]
    # 更新概率
    for cur_round in range(all_round_num):
        for card_ec ,action_tree in round_card_reg_tree[cur_round].items():
            for node ,reg in action_tree.items():
                # CFR
                reg = [max(i,0) for i in reg]
                action_p  = get_save_p(reg)
                round_card_action_tree[cur_round][card_ec][node] = action_p
                # CFR plus
                action_tree[node] = reg

    # for k,d in action_sump_tree_dict


def train(msg):
    player_poker, table_poker = msg
    data = [get_action_tree(player_poker[i], table_poker) for i in range(2)]
    action_p_tree_list = [i[0] for i in data]
    encode_node_list = [i[1] for i in data]
    get_reg_tree(player_poker, table_poker,action_p_tree_list ,encode_node_list)
    return None
#
# f1 = open('data/train_data_base.npy', 'wb')
# pickle.dump([round_card_action_tree,round_card_reg_tree,round_card_sump_tree ], f1)
# f1.close()
save_num = [10, 1000, 10000]
save_n = 10001
train_n = 10000
for train_step in range(save_n):
    for _ in range(train_n):
        train(mc_deal_pokers(res_poker))
    if train_step in save_num:
        f1 = open(f'../data/train_data_mc_p{train_step}w.npy', 'wb')
        pickle.dump([round_card_action_tree, round_card_reg_tree, round_card_sump_tree], f1)
        f1.close()
    # reg_list = list(map(train,deal_pokers(res_poker)))

    # for player_poker,table_poker in deal_pokers():
    #     card_169_list, reg_tree, sum_action_p = train(player_poker,table_poker)
    #     re
    # list(map(train, deal_pokers(res_poker)))
    # pool = Pool()
    # reg_list = pool.map(train, deal_pokers(res_poker))
    # pool.close()
    # pool.join()
    # for msg in deal_pokers(res_poker):
    #     train(msg)+
    #     break
    # reg_list  =
    # print(time.time()-s)
    # update()
    # print(time.time()-s)
    # del reg_list
    # f1 = open('data/train_datax'+str(train_step)+'.npy', 'wb')
    # pickle.dump([round_card_action_tree,round_card_reg_tree,round_card_sump_tree ], f1)
    # f1.close()
# #
# print(data2['3661126667'])
# print(data2.keys() )
# data2 = np.load("End_game_tree.npy",allow_pickle=True)
# print(data2.keys()[:2])

# round_card_action_tree,round_card_reg_tree,round_card_sump_tree
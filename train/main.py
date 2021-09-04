import numpy as np
import pandas as pd
import time
import argparse
from br import BRAgent
from cfr import CFRAgent
from xdo import XDOAgent
from oxdo import OXDOAgent
from mccfr import MCCFRAgent

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-g','--game', help='game', default="leduc", required=True)
parser.add_argument('-a','--algo', help='algorithms', default="cfr", required=True)
args = vars(parser.parse_args())

GAME = args['game']
ALGO = args['algo']

if GAME == "kuhn":
    from utils_kuhn import encode_cards, fold_encode, deck, get_winner, round_card_num, deal_pokers
elif GAME == "leduc" or GAME == "leduc_clone":
    from utils_leduc import encode_cards, fold_encode, deck, get_winner, round_card_num, deal_pokers

xdo_agent = XDOAgent(GAME, encode_cards, fold_encode, deck, get_winner, round_card_num, deal_pokers)
cfr_agent = CFRAgent(GAME, encode_cards, fold_encode, deck, get_winner, round_card_num, deal_pokers)
br_agent = BRAgent(GAME, encode_cards, fold_encode, deck, get_winner, round_card_num, deal_pokers)
oxdo_agent = OXDOAgent(GAME, encode_cards, fold_encode, deck, get_winner, round_card_num, deal_pokers)
mccfr_agent = OXDOAgent(GAME, encode_cards, fold_encode, deck, get_winner, round_card_num, deal_pokers)


process_num = 20
testing_num = 1000
training_times = 500
test_adjunct = 1


def train_and_evaluate():
    print(f"start training {ALGO} on {GAME}")
    start = time.time()
    results, times = [], []
    policy = {}
    for train_step in range(training_times):
        print('train', train_step)
        if train_step % test_adjunct == 0:
            # multi_test([cfr_agent, random_agent], process_num=process_num, testing_num=testing_num, game=cfr_agent.game)
            exp = br_agent.exploitability(policy)
            print(time.time() - start, exp)
            if exp < 0:
                raise Exception("Exploitability negative")
            times.append(time.time() - start)
            results.append(exp)

        if ALGO == 'xdo':
            policy = xdo_agent.train(
                cfr_iterations=20000,
                exp_threshold=2 ** 4,
                check_period=50
            )
        elif ALGO == 'oxdo':
            policy = oxdo_agent.train()
        elif ALGO == 'cfr':
            policy = cfr_agent.train()
        elif ALGO == 'mccfr':
            policy = mccfr_agent.train()
        else:
            policy = None

    print("Done")
    pd.DataFrame({"times": times, "exp": results}).to_csv(f"../plot/{ALGO}_{GAME}_exploitability")


train_and_evaluate()

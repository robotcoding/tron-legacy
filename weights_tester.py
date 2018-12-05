import time
from tronproblem import TronProblem
import copy, argparse, signal
from collections import defaultdict
import support
import random
import numpy as np
from gamerunner import run_game

def main():
    empty_game = TronProblem("maps/empty_room.txt", 0)
    joust_game = TronProblem("maps/joust.txt", 0)
    hunger_game = TronProblem("maps/hunger_games.txt", 0)
    divider_game = TronProblem("maps/divider.txt", 0)

    bots = support.determine_bot_functions(["ta2", "student"])

    e_total = 0
    j_total = 0
    h_total = 0
    d_total = 0

    for trials in range(100):
        e_total = e_total + run_game(copy.deepcopy(empty_game), bots).index(1)
        j_total = j_total + run_game(copy.deepcopy(joust_game), bots).index(1)
        h_total = h_total + run_game(copy.deepcopy(hunger_game), bots).index(1)
        d_total = d_total + run_game(copy.deepcopy(divider_game), bots).index(1)


    print("Empty: {} Joust: {} HG: {} Divider: {}\n".format(e_total, j_total, h_total, d_total))

if __name__ == "__main__":
    main()

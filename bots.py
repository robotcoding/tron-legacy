#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math

# Throughout this file, ASP means adversarial search problem.


class StudentBot:
    """ Write your student bot here"""

    def voronoi_boi(self, state):
        board_partitions = [0 for i in range(3)]
        for i in range(len(state.board)):
            for j in range(len(state.board[0])):
                # Skip over permanent and temporary barriers
                if state.board[i][j] == '#' or state.board[i][j] == 'x':
                    continue

                # Calculate each player's distance from (i, j)
                one_dist = abs(i - state.player_locs[0][0]) + abs(j - state.player_locs[0][1])
                two_dist = abs(i - state.player_locs[1][0]) + abs(j - state.player_locs[1][1])
                if one_dist > two_dist: # If player one is closer
                    board_partitions[0] += 1
                elif one_dist < two_dist: # If player two is closer
                    board_partitions[1] += 1
                else: # If there's a tie
                    board_partitions[2] += 1
        return board_partitions

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        """
        state = asp.get_start_state()
        locs = state.player_locs
        ptm = state.ptm
        loc = locs[ptm]
        actions = list(asp.get_safe_actions(state.board, loc))
        next_states = [asp.transition(state, a) for a in actions]
        voronois = [self.voronoi_boi(s) for s in next_states]
        dists = [voronois[i][ptm] for i in range(len(voronois))]
        max_index = 0
        for j in range(len(dists)):
            if dists[j] > dists[max_index]:
                max_index = j
        return actions[max_index]

    def cleanup(self):
        """
        Input: None
        Output: None

        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """
        pass


class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        pass


class WallBot:
    """Hugs the wall"""

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"
        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision

#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math

# Throughout this file, ASP means adversarial search problem.


class StudentBot:
    """ Write your student bot here"""

    def divide_empty_territory(self, board):
        spaces_to_comp = {}
        comps = -1
        for i in range(len(board) - 1):
            for j in range(len(board[0]) - 1):
                # Skip over permanent and temporary barriers
                if board[i][j] == '#' or board[i][j] == 'x':
                    continue

                if not (i, j) in spaces_to_comp.keys():
                    comps +=1
                    spaces_to_comp[(i, j)] = comps

                if not (board[i+1][j] == '#' or board[i+1][j] == 'x'):
                    spaces_to_comp[(i+1, j)] = spaces_to_comp[(i, j)]

                if not (board[i][j+1] == '#' or board[i][j+1] == 'x'):
                    spaces_to_comp[(i, j+1)] = spaces_to_comp[(i, j)]

        return spaces_to_comp

    def voronoi_boi(self, state):
        board_partitions = [0 for i in range(3)]
        spaces_to_comp = self.divide_empty_territory(state.board)
        one_comp = spaces_to_comp[state.player_locs[0]]
        two_comp = spaces_to_comp[state.player_locs[1]]
        one_loc = state.player_locs[0]
        two_loc = state.player_locs[1]
        for i in range(len(state.board)):
            for j in range(len(state.board[0])):
                # Skip over permanent and temporary barriers
                if state.board[i][j] == '#' or state.board[i][j] == 'x':
                    continue

                if spaces_to_comp[(i, j)] == one_comp:
                    if spaces_to_comp[(i, j)] != two_comp:
                        board_partitions[0] += 1
                    else:
                        # Calculate each player's distance from (i, j)
                        one_dist = abs(i - state.player_locs[0][0]) + abs(j - state.player_locs[0][1])
                        two_dist = abs(i - state.player_locs[1][0]) + abs(j - state.player_locs[1][1])
                        if one_dist > two_dist: # If player one is closer
                            board_partitions[0] += 1
                        elif one_dist < two_dist: # If player two is closer
                            board_partitions[1] += 1
                        else: # If there's a tie
                            board_partitions[2] += 1
                elif spaces_to_comp[(i, j)] == two_comp:
                    board_partitions[1] += 1
        '''
        offsets = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        one_surrounds = [(one_loc[0]+offset[0], one_loc[1]+offset[1]) for offset in offsets]
        walls_by_one = 0.0
        for loc in one_surrounds:
            if state.board[loc[0]][loc[1]] == '#' or state.board[loc[0]][loc[1]] == 'x':
                walls_by_one += 1

        two_surrounds = [(two_loc[0]+offset[0], two_loc[1]+offset[1]) for offset in offsets]
        walls_by_two = 0.0
        for loc in two_surrounds:
            if state.board[loc[0]][loc[1]] == '#' or state.board[loc[0]][loc[1]] == 'x':
                walls_by_two += 1


        board_partitions[0] *= walls_by_one
        board_partitions[1] *= walls_by_two
        '''
        if not one_comp == two_comp:
            if board_partitions[0] > board_partitions[1]:
                board_partitions[0] *= 3
            elif board_partitions[0] < board_partitions[1]:
                board_partitions[1] *= 3

        return board_partitions

    def alpha_beta_cutoff(self, asp):
        # Initialize the variables
        state = asp.get_start_state()
        me = state.player_to_move()
        # No action can be taken from a terminal state
        if asp.is_terminal_state(state):
            return None
        else:
            # Use None in place of positive and negative infinity
            return self.ab_cutoff_helper(asp, state, None, None, 0, me)[0]

    def ab_cutoff_helper(self, asp, state, a, b, depth, me):
        look_ahead_cutoff = 5
        if asp.is_terminal_state(state):
            # Return the value
            if state.player_locs[me] == None:
                return (None, -99999)
            else:
                return (None, 99999)
        else:
            # Recurse to find the node's value
            locs = state.player_locs
            ptm = state.ptm
            loc = locs[ptm]
            actions = list(asp.get_safe_actions(state.board, loc))
            if len(actions) == 0:
                if ptm == me:
                    return ('U', -99999)
                else:
                    return ('U', 99999)
            initialized = False
            for action in actions:
                # If this state is as deep as we can go
                if depth == look_ahead_cutoff: # Note that this assumes that we're the ptm here
                    # Evaluate it with our function
                    board_partition = self.voronoi_boi(state)
                    #return (None, board_partition[me] - board_partition[1-me])
                    return (None, board_partition[me])
                else:
                    # Get the next state and find its value
                    new_state = asp.transition(state, action)
                    value = self.ab_cutoff_helper(asp, new_state, a, b, depth + 1, me)
                    # If this is our first action
                    if not initialized:
                        # It is the best action
                        best = (action, value[1])
                        initialized = True
                    if ptm == me:
                        # Maximize and adjust alpha
                        if value[1] > best[1]:
                            best = (action, value[1])
                        # If alpha is initialized
                        if (not a == None):
                            # Take the max
                            a = max(a, value[1])
                        else:
                            # Otherwise just use the value
                            a = value[1]
                    else:
                        # Minimize and adjust beta
                        if value[1] < best[1]:
                            best = (action, value[1])
                        # If beta is initialized
                        if (not b == None):
                            # Take the min
                            b = min(b, value[1])
                        else:
                            # Otherwise just use the value
                            b = value[1]
                    # Prune the tree
                    if (not a == None) and (not b == None) and a >= b:
                        break

            return best

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        """
        return self.alpha_beta_cutoff(asp)
        '''
        state = asp.get_start_state()
        locs = state.player_locs
        ptm = state.ptm
        loc = locs[ptm]
        actions = list(asp.get_safe_actions(state.board, loc))
        if len(actions) == 0:
            return "U"
        next_states = [asp.transition(state, a) for a in actions]
        voronois = [self.voronoi_boi(s) for s in next_states]
        dists = [voronois[i][ptm] for i in range(len(voronois))]
        max_index = 0
        for j in range(len(dists)):
            if dists[j] > dists[max_index]:
                max_index = j
        return actions[max_index]
        '''

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

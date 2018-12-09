#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math
import copy
import heapq

# Throughout this file, ASP means adversarial search problem.
ART_WT = 0.25

class StudentBot:
    """ Write your student bot here"""

    def __init__(self):
        self.next_to_powerup = False
        self.dir_to_powerup = ''
        self.BOT_NAME = "goddard"

    # def get_neighbors(self, board, loc):
    #     r = []
    #     x = loc[0]
    #     y = loc[1]
    #
    #     up = (x, y+1)
    #     down = (x, y-1)
    #     right = (x+1, y)
    #     left = (x-1, y)
    #
    #     possible = [up, down, right, left]
    #     for x0, y0 in possible:
    #         if x0 >= 0 and x0 < len(board[0]) and y0 >= 0 and y0 < len(board):
    #             r.append((x0, y0))
    #
    #     return r
    #
    # # takes board and loc xy tuple, returns length of shortest path
    # def dijkstra(self, board, src, dst):
    #     #q = []
    #     q_src = []
    #     q_dst = []
    #     dist_src = copy.deepcopy(board)
    #     dist_dst = copy.deepcopy(board)
    #     #prev = copy.deepcopy(board)
    #     for i in range(len(board)):
    #         for j in range(len(board[0])):
    #             val = float("inf")
    #             dist[i][j] = val
    #             #prev[i][j] = None
    #             #q.append((i,j)) # add all xy tuples to queue
    #             heapq.heappush(q, (val, (i, j)))
    #             heapq.heappush(q, (val, (i, j)))
    #     dist_src[src[0]][src[1]] = 0
    #     dist_dst[dst[0]][dst[1]] = 0
    #     while not len(q_src) == 0 and not len(q_dst) == 0:
    #         if not len(q_src) == 0:
    #             # print(len(q))
    #             #u = min(q, key=lambda v:  dist[v[0]][v[1]])
    #             u_dst, u_coor = heapq.heappop(q_src)
    #             #q.remove(u)
    #             if u_coor == ds or dist_dst[u_coor[0]][u_coor[1]] != float("inf")t:
    #                 break # either it's an overlap or you're at beginning
    #             neighbors = self.get_neighbors(board, src)
    #
    #             for v in neighbors:
    #                 alt_dist = dist_src[u_coor[0]][u_coor[1]] + 1
    #                 if alt_dist < dist_src[v[0]][v[1]]:
    #                     dist_src[v[0]][v[1]] = alt_dist
    #                     #prev[v[0]][v[1]] = u
    #         if not len(q_dst) == 0:
    #             # print(len(q))
    #             #u = min(q, key=lambda v:  dist[v[0]][v[1]])
    #             u_dst, u_coor = heapq.heappop(q_dst)
    #             #q.remove(u)
    #             if u_coor == dst or dist_src[u_coor[0]][u_coor[1]] != float("inf"):
    #                 break
    #             neighbors = self.get_neighbors(board, dst)
    #
    #             for v in neighbors:
    #                 alt_dist = dist_dst[u_coor[0]][u_coor[1]] + 1
    #                 if alt_dist < dist_dst[v[0]][v[1]]:
    #                     dist_dst[v[0]][v[1]] = alt_dist
    #                     #prev[v[0]][v[1]] = u
    #     #return dist[dst[0]][dst[1]]
    #     return dist_src[dst[0]][dst[1]]

    def is_art_point(self, my_loc, board, comps):
        board_copy = copy.deepcopy(board)
        board_copy[my_loc[0]][my_loc[1]] = "x"
        divided = self.divide_empty_territory(board_copy)
        return divided[1] > comps

    def dist(self, x0, y0, x1, y1):
        return abs(x0 - x1) + abs(y0 - y1)

    def has_the_power(self, board, i, j):
        had = board[i][j] == "*" or board[i][j] == "@" or board[i][j] == "^" or board[i][j] == "!"
        #print(had)
        return had

    # based on a board and a loc (xy tuple), finds powerups closest to loc
    # returns list of powerup coordinates and their respective distance from my_loc
    #       sorted by distance
    #       [(x0,y0,dist0), (x1, y1, d1), ...]
    def find_powerups(self, my_loc, board):
        locs = []
        for i in range(len(board) - 1):
            for j in range(len(board[0]) - 1):
                if board[i][j] == "*" or board[i][j] == "@" or board[i][j] == "^" or board[i][j] == "!":
                    locs.append((i, j, self.dist(i, j, my_loc[0], my_loc[1]))) # append tuple of xy loc
        locs.sort(key=lambda x:x[2])
        return locs # based on this determine a direction to prioritize?

    # based on a state and a player loc in that state, finds how many of their 4
    # surrounding cells (U, D, L, R) correspond to barriers
    # returns number of such walls as a value b/w 0 and 4
    def get_wall_val(self, state, loc):
        wall_value = 0
        offsets = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        my_surrounds = [(loc[0]+offset[0], loc[1]+offset[1]) for offset in offsets]
        for l in my_surrounds:
            if state.board[l[0]][l[1]] == '#' or state.board[l[0]][l[1]] == 'x':
                wall_value += 1
        return wall_value

    # based on a board, determines which "component" each cell belongs to, where
    # a component is defined as a space which is entirely enclosed by barriers.
    # returns a tuple of: dictionary mapping each cell that does not contain a barrier to
    # some component index, and the # of components in that dictionary
    def divide_empty_territory(self, board):
        spaces_to_comp = {}
        comps = -1
        for i in range(len(board) - 1):
            for j in range(len(board[0]) - 1):
                # Skip over permanent and temporary barriers
                if board[i][j] == '#' or board[i][j] == 'x':
                    continue

                if (i,j+1) in spaces_to_comp.keys():
                    spaces_to_comp[(i, j)] = spaces_to_comp[(i, j+1)]

                if not (i, j) in spaces_to_comp.keys():
                    comps += 1
                    spaces_to_comp[(i, j)] = comps

                if not (board[i+1][j] == '#' or board[i+1][j] == 'x'):
                    spaces_to_comp[(i+1, j)] = spaces_to_comp[(i, j)]

                if not (board[i][j+1] == '#' or board[i][j+1] == 'x'):
                    spaces_to_comp[(i, j+1)] = spaces_to_comp[(i, j)]

        return (spaces_to_comp, comps)

    # given a state, a components dictionary, location of player 1 and location
    # of player 2, calculates board "partitions" dividing board into three
    # regions: ones closer to player1, ones closer to player 2, and those
    # equidistant from both players
    def voronoi_boi(self, state, components, one_loc, two_loc):
        # Initialize variables
        board_partitions = [0 for i in range(3)]
        one_comp = components[one_loc]
        two_comp = components[two_loc]

        # Iterate through board
        for i in range(1, len(state.board)):
            for j in range(1, len(state.board[0])):
                # Skip over permanent and temporary barriers
                if state.board[i][j] == '#' or state.board[i][j] == 'x':
                    continue

                # Extra count that helps it wall follow
                # Basically values us having open space without walls in middle
                open_space = 0 #
                if one_comp == two_comp:
                    open_space = self.get_wall_val(state, (i, j))

                # If player one can access the space
                if components[(i, j)] == one_comp:
                    if components[(i, j)] != two_comp:
                        # Give it to player one if player two can't
                        board_partitions[0] += 1 + open_space
                    else:
                        # Calculate each player's distance from (i, j)
                        # one_path = self.dijkstra(board, one_loc)
                        # two_path = self.dijkstra(board, two_loc)
                        # one_dist = len(one_path)
                        # two_dist = len(two_path)
                        # one_dist = self.dijkstra(state.board, one_loc, (i,j))
                        # two_dist = self.dijkstra(state.board, two_loc, (i,j))
                        one_dist = self.dist(i, j, one_loc[0], one_loc[1])
                        two_dist = self.dist(i, j, two_loc[0], two_loc[1])
                        if one_dist < two_dist: # If player one is closer
                            board_partitions[0] += 1 + open_space
                        elif one_dist > two_dist: # If player two is closer
                            board_partitions[1] += 1 + open_space
                        else: # If there's a tie
                            board_partitions[2] += 1 + open_space
                # If only player two can access it
                elif components[(i, j)] == two_comp:
                    board_partitions[1] += 1 + open_space

        return board_partitions

    # given a state and an action to be taken by player "me":
    # calculates a value based on weighing benefits of gaining territory,
    # getting powerups, hugging walls
    # returns an integer

    def evaluate_state(self, last_state, state, action, me):
        comp_res = self.divide_empty_territory(state.board)
        spaces_to_comp = comp_res[0]
        comps = comp_res[1]
        my_loc = state.player_locs[me]
        opp_loc = state.player_locs[1-me]
        my_comp = spaces_to_comp[my_loc]
        opp_comp = spaces_to_comp[opp_loc]

        closest_spaces = self.voronoi_boi(state, spaces_to_comp, my_loc, opp_loc)
        powerups = self.find_powerups(my_loc, state.board)

        territory_value = 0
        space_value = 0
        wall_value = 0
        powerup_value = 0
        territory_weight = 40
        space_weight = 50
        wall_weight = 20
        powerup_weight = 10
        articulation_weight = 1

        # The territory is a heuristic to make our bot try to claim space
        territory_value = closest_spaces[0] - closest_spaces[1]

        # Value of how many spaces are closest to us, not difference, again to claim space
        space_value = closest_spaces[0]

        # How many spaces around this position are walls, to hug walls
        wall_value = self.get_wall_val(state, my_loc)

        ### STILL HAVE NO CLUE HOW TO MOTIVATE IT TO TAKE POWERUPS, THIS DOESN'T WORK ###
        ### DOES IT MATTER THOUGH IF IT WINS? ###
        if not last_state == None:
            if self.has_the_power(last_state.board, my_loc[0], my_loc[1]):
                powerup_value = 50

        '''
        if self.next_to_powerup and self.dir_to_powerup == action:
            powerup_value = 30

        self.next_to_powerup = False
        self.dir_to_powerup = ''
        '''
        # Motivate us to move in the direction of powerups
        if not powerups == []: # If there are powerups
            goal = powerups[0]
            if goal[0] < my_loc[0] and action == 'U':
                powerup_value = 5
                #if goal[2] == 1:
                    #self.next_to_powerup = True
                    #self.dir_to_powerup = 'U'
            elif goal[0] > my_loc[0] and action == 'D':
                powerup_value = 5
                #if goal[2] == 1:
                    #self.next_to_powerup = True
                    #self.dir_to_powerup = 'D'
            elif goal[1] < my_loc[1] and action == 'L':
                powerup_value = 5
                #if goal[2] == 1:
                    #self.next_to_powerup = True
                    #self.dir_to_powerup = 'L'
            elif goal[1] > my_loc[1] and action == 'R':
                powerup_value = 5
                #if goal[2] == 1:
                    #self.next_to_powerup = True
                    #self.dir_to_powerup = 'R'
        else:
            powerup_weight = 0

        if not my_comp == opp_comp: # If we're in separate components
            # We're in the end game and need to wall-hug
            territory_weight = 1000
            space_weight = 100
            wall_weight = 30
            powerup_weight = 5
            if self.is_art_point(my_loc, state.board, comps):
                articulation_weight = ART_WT

        return articulation_weight * (territory_value * territory_weight + space_value * space_weight + wall_value * wall_weight + powerup_value * powerup_weight)

    def is_unsafe_but_ok(self, state, loc, player):
        board = state.board
        if state.player_has_armor(player):
            # yee
            wall_val = self.get_wall_val(state, loc)
            if wall_val < 3:
                if wall_val == 2:
                    opp_char = str(1-player + 1)
                    offsets = [(-1, 0), (1, 0), (0, 1), (0, -1)]
                    my_surrounds = [(loc[0]+offset[0], loc[1]+offset[1]) for offset in offsets]
                    for l in my_surrounds:
                        if state.board[l[0]][l[1]] == opp_char:
                            return False
                return True
        return False
    # returns best action to take
    def alpha_beta_cutoff(self, asp):
        # Initialize the variables
        state = asp.get_start_state()
        me = state.player_to_move()
        # No action can be taken from a terminal state
        if asp.is_terminal_state(state):
            return None
        else:
            # Use None in place of positive and negative infinity
            return self.ab_cutoff_helper(asp, None, state, None, None, 0, me)[0]

    # returns tuple of best action to take and its value
    def ab_cutoff_helper(self, asp, last_state, state, a, b, depth, me):
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
            safe_actions_set = asp.get_safe_actions(state.board, loc)
            all_actions_set = asp.get_available_actions(state)
            only_unsafe = all_actions_set.difference(safe_actions_set)
            actions = list(asp.get_safe_actions(state.board, loc))
            unsafe_but_safe = [x for x in only_unsafe if self.is_unsafe_but_ok(state, loc, ptm)]
            #unsafe_but_safe = [x for x in only_unsafe if self.is_unsafe_but_ok(state.transition(state, x), loc, ptm)]
            actions += unsafe_but_safe
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
                    evalu = self.evaluate_state(last_state, state, action, me)
                    #return (None, board_partition[me] - board_partition[1-me])
                    return (None, evalu) # where is the direction of choice passed back?
                else:
                    # Get the next state and find its value
                    new_state = asp.transition(state, action)
                    value = self.ab_cutoff_helper(asp, state, new_state, a, b, depth + 1, me)
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

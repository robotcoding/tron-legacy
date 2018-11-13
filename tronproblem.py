from adversarialsearchproblem import AdversarialSearchProblem, GameState
from boardprinter import BoardPrinter
import random
from trontypes import CellType, PowerupType
import copy


class TronState(GameState):
    def __init__(self, board, player_locs, ptm, player_powerups):
        """
        Input:
            board- a list of lists of characters representing cells
                ('#' for wall, ' ' for space, etc.)
            player_locs- a list of tuples (representing the players' locations)
            ptm- the player whose move it is. player_locs and ptm are
                indexed the same way, so player_locs[ptm] would
                give the location of the player whose move it is.
            player_powerups- a map from player to a map of what powerups they have
                {player : {PowerupType : powerup value}}
        """
        self.board = board
        self.player_locs = player_locs
        self.ptm = ptm
        self.player_powerups = player_powerups

    def player_to_move(self):
        return self.ptm

    def player_has_armor(self, player):
        """
        Input:
            player- the zero-indexed number representing the player
        Output:
            true if the player has armor active, false otherwise
        """
        assert player in self.player_powerups
        return PowerupType.ARMOR in self.player_powerups[player]

    def get_remaining_turns_speed(self, player):
        """
        Input:
            player- the zero-indexed number representing the player
        Output:
            the number of turns remaining from the speed powerup.
            if no turns are remaining, returns 0
        """
        assert player in self.player_powerups
        if PowerupType.SPEED in self.player_powerups[player]:
            return (self.player_powerups[player])[PowerupType.SPEED]
        return 0


TRAP_QUANTITY = 3  # number of barriers placed by trap powerup
BOMB_RADIUS = 4  # radius of barriers removed by bomb powerup
SPEED_BOOST = 4  # number of consecutive turns given with speed powerup

# directions to move
U = "U"
D = "D"
L = "L"
R = "R"


class TronProblem(AdversarialSearchProblem):
    def __init__(self, board_file_loc, first_player):
        """
        Initializes the tronproblem.
        You won't need to call this directly if you use gamerunner
        Input:
            board_file_loc- location of board (map) file
            first_player- the first player to move
        """
        board = TronProblem._board_from_board_file(board_file_loc)
        player_locs = TronProblem._player_locs_from_board(board)

        player_powerups = {}
        for i in range(len(player_locs)):
            player_powerups[i] = {}

        self._start_state = TronState(board, player_locs, first_player, player_powerups)
        self._num_players = len(player_locs)

    ###### ADVERSARIAL SEARCH PROBLEM IMPLEMENTATION ######

    def get_available_actions(self, state):
        """
        Returns all moves (even moves that would result in immediate collisions)
        Use get_safe_actions if you want all moves that won't be an immediate collision

        We assume that the player to move is never on the edges of the map.
        All pre-made maps are surrounded by walls to validate this assumption.
        """
        return {U, D, L, R}

    def transition(self, state, action):
        assert not (self.is_terminal_state(state))
        assert action in self.get_available_actions(state)

        # prepare parts of result state
        board = [[elt for elt in row] for row in state.board]
        player_locs = [loc for loc in state.player_locs]
        next_ptm = (state.ptm + 1) % self._num_players
        player_powerups = copy.deepcopy(state.player_powerups)
        while player_locs[next_ptm] == None:
            next_ptm = (next_ptm + 1) % self._num_players
        # note that, given the assumption that state is non-terminal,
        # there will be at least 2 players still on the board when
        # going through this loop.

        # get original position of player to move before transitioning
        r0, c0 = state.player_locs[state.ptm]

        # lay down a barrier where the player was before
        board[r0][c0] = CellType.BARRIER

        # get target location after moving
        r1, c1 = TronProblem.move((r0, c0), action)

        # resolve move based on the cell that the player tried to move to
        cell = state.board[r1][c1]
        if cell == CellType.SPACE:  # move to empty space
            TronProblem._move_player_and_update(board, state, player_locs, r1, c1)

        elif cell == CellType.TRAP:  # move to trap powerup
            TronProblem._move_player_and_update(board, state, player_locs, r1, c1)

            # place barriers in front of next player
            board = TronProblem._add_barriers(board, player_locs[next_ptm])

        elif cell == CellType.BOMB:  # move to bomb powerup
            TronProblem._move_player_and_update(board, state, player_locs, r1, c1)

            # remove barriers around current player
            board = TronProblem._remove_barriers(board, player_locs[state.ptm])

        elif cell == CellType.ARMOR:  # move to armor powerup
            # fill new space with player symbols and update
            TronProblem._move_player_and_update(board, state, player_locs, r1, c1)

            # give current player armor powerup
            TronProblem._add_powerup(
                state.ptm, player_powerups, PowerupType.ARMOR, 1
            )

        elif cell == CellType.SPEED:  # move to speed powerup
            TronProblem._move_player_and_update(board, state, player_locs, r1, c1)

            # give current player speed powerup
            TronProblem._add_powerup(
                state.ptm, player_powerups, PowerupType.SPEED, SPEED_BOOST
            )

        else:  # player chose to move into an occupied space.
            # if they have armor and the space is a barrier,
            # move normally into the space and remove the armor
            if state.player_has_armor(state.ptm) and cell == CellType.BARRIER:
                TronProblem._move_player_and_update(board, state, player_locs, r1, c1)

                player_powerups[state.ptm].pop(
                    PowerupType.ARMOR, None
                )  # remove armor
            else:
                # otherwise, they crashed so they are removed from the game
                player_locs[state.ptm] = None

        # if the player has the speed powerup, set the player-to-move
        # of the next state to be the current player-to-move and decrement
        # the speed counter
        if state.get_remaining_turns_speed(state.ptm) > 0:
            if (player_powerups[state.ptm])[PowerupType.SPEED] <= 1:
                player_powerups[state.ptm].pop(PowerupType.SPEED, None)
            else:
                (player_powerups[state.ptm])[PowerupType.SPEED] -= 1
            # return state with the same player moving as current turn
            return TronState(board, player_locs, state.ptm, player_powerups)

        # return the next state
        return TronState(board, player_locs, next_ptm, player_powerups)

    def is_terminal_state(self, state):
        num_players_left = 0
        for pl in state.player_locs:
            if not (pl == None):
                num_players_left += 1

        return num_players_left == 1

    def evaluate_state(self, state):
        """
        Note that, since players take turns sequentially,
        ties are impossible.
        """
        assert self.is_terminal_state(state)

        values = [0.0 if pl == None else 1 for pl in state.player_locs]
        return values

    ###### STATIC METHODS FOR IMPLEMENTING METHODS ABOVE ######

    @staticmethod
    def _add_barriers(board, loc):
        """
        adds barriers around loc as specified by the handout
        Input:
            board- a list of lists of characters representing cells
            loc- location to center the added barriers
        """
        rows = len(board)
        cols = len(board[0])
        r, c = loc
        valid = []

        for i in range(-2, 3):
            for j in range(-2, 3):
                if r + i >= 0 and r + i < rows:
                    if c + j >= 0 and c + j < cols:
                        if board[r + i][c + j] == CellType.SPACE:
                            if abs(i) == 2 or abs(j) == 2:
                                valid.append((r + i, c + j))

        random.shuffle(valid)
        to_place = TRAP_QUANTITY
        while to_place > 0 and valid:
            row, col = valid.pop()
            board[row][col] = CellType.BARRIER  # place a barrier
            to_place -= 1

        return board

    @staticmethod
    def _remove_barriers(board, loc):
        """
        removes barriers around loc as specified by the handout
        Input:
            board- a list of lists of characters representing cells
            loc- location to center the added barriers
        """
        rows = len(board)
        cols = len(board[0])
        r, c = loc

        for i in range(-BOMB_RADIUS, BOMB_RADIUS + 1):
            for j in range(-BOMB_RADIUS, BOMB_RADIUS + 1):
                if r + i >= 0 and r + i < rows:
                    if c + j >= 0 and c + j < cols:
                        if board[r + i][c + j] == CellType.BARRIER:
                            board[r + i][c + j] = CellType.SPACE  # remove barrier
        return board

    @staticmethod
    def _move_player_and_update(board, state, player_locs, r1, c1):
        """
        adds player location to map, then stores the player
        location in player_locs
        """
        board[r1][c1] = str(state.ptm + 1)  # add player location to map
        player_locs[state.ptm] = (r1, c1)  # stores player location

    @staticmethod
    def _board_from_board_file(board_file_loc):
        board_file = open(board_file_loc)
        board = []
        for line in board_file.readlines():
            line = line.strip()
            row = [
                random.choice(CellType.powerup_list) if c is "?" else c
                for c in line
                if not (c == "\n")
            ]
            board.append(row)
        return board

    @staticmethod
    def _player_locs_from_board(board):
        loc_dict = {}
        for r in range(len(board)):
            for c in range(len(board[r])):
                char = board[r][c]
                if TronProblem._is_int(char):
                    index = int(char) - 1
                    loc_dict[index] = (r, c)

        loc_list = []
        num_players = len(loc_dict)
        for index in range(num_players):
            loc_list.append(loc_dict[index])
        return loc_list

    @staticmethod
    def _add_powerup(player, player_powerups, powerup, value):
        assert player in player_powerups
        (player_powerups[player])[powerup] = value

    @staticmethod
    def _is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def move(loc, direction):
        """
        Produces the location attained by going in the given direction
        from the given location.

        loc will be a (<row>, <column>) double, and direction will be
        U, L, D, or R.
        """
        r0, c0 = loc
        if direction == U:
            return (r0 - 1, c0)
        elif direction == D:
            return (r0 + 1, c0)
        elif direction == L:
            return (r0, c0 - 1)
        elif direction == R:
            return (r0, c0 + 1)
        else:
            raise ValueError("The input direction is not valid.")

    ###### HELPFUL FUNCTIONS FOR YOU ######

    @staticmethod
    def is_cell_player(board, loc):
        """
        Input:
            board- a list of lists of characters representing cells
            loc- location (<row>, <column>) on the board
        Output:
            Returns true if the cell at loc is a player, which is true when
            the player is a digit, or false otherwise.
        """
        r, c = loc
        return board[r][c].isdigit()

    @staticmethod
    def get_safe_actions(board, loc):
        """
        Given a game board and a location on that board,
        returns the set of actions that don't result in immediate collisions.
        Input:
            board- a list of lists of characters representing cells
            loc- location (<row>, <column>) to find safe actions from
        Output:
            returns the set of actions that don't result in immediate collisions.
            An immediate collision occurs when you run into a barrier, wall, or
            the other player
        """
        safe = set()
        for action in {U, D, L, R}:
            r1, c1 = TronProblem.move(loc, action)
            if not (
                board[r1][c1] == CellType.BARRIER
                or board[r1][c1] == CellType.WALL
                or TronProblem.is_cell_player(board, (r1, c1))
            ):
                safe.add(action)
        return safe

    @staticmethod
    def visualize_state(state, colored):
        print(BoardPrinter.state_to_string(state, colored))

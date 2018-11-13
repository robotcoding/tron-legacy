#!/usr/bin/python

from trontypes import CellType, PowerupType


class TextColors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


color_dict = {
    CellType.ARMOR: TextColors.BLUE,
    CellType.BOMB: TextColors.RED,
    CellType.TRAP: TextColors.YELLOW,
    CellType.SPEED: TextColors.GREEN,
    CellType.WALL: TextColors.BOLD,
}


class BoardPrinter:
    @staticmethod
    def state_to_string(state, colored):
        """
        Input:
            state- TronState to stringify
            colored- boolean. if true, use color
        Output:
            Returns a string representing a readable version of the state.
            First part of the string is the powerup description,
            second part is the actual board
        """
        if colored:
            return "{}{}".format(
                BoardPrinter._powerup_description(state),
                BoardPrinter._board_to_pretty_string_colored(state),
            )
        else:
            return BoardPrinter._powerup_description(
                state
            ) + BoardPrinter._board_to_pretty_string(state.board)

    @staticmethod
    def _board_to_pretty_string(board):
        s = ""
        for row in board:
            for cell in row:
                s += cell
            s += "\n"
        return s

    @staticmethod
    def _colored_character(cell, state):
        color = None
        if cell in color_dict:
            color = color_dict[cell]
        elif cell.isdigit() and BoardPrinter._is_any_player_speeding(state):
            color = color_dict[CellType.SPEED]
        elif cell.isdigit() and state.player_has_armor(int(cell) - 1):
            color = color_dict[CellType.ARMOR]
        else:
            return cell
        return "{}{}{}".format(color, cell, TextColors.ENDC) if color else cell

    @staticmethod
    def _is_any_player_speeding(state):
        for k, v in state.player_powerups.items():
            if PowerupType.SPEED in v:
                return True
        return False

    @staticmethod
    def _board_to_pretty_string_colored(state):
        s = ""
        for row in state.board:
            for cell in row:
                s += BoardPrinter._colored_character(cell, state)
            s += "\n"
        return s

    @staticmethod
    def _powerup_description(state):
        s = ""
        for i in range(len(state.player_locs)):
            s += "Player {}: {}\n".format(
                i + 1, "Armor" if state.player_has_armor(i) else "..."
            )
        return s

import time
from tronproblem import TronProblem
import copy, argparse, signal
from collections import defaultdict
import support
import random
import numpy as np


def run_game(asp, bots, visualizer=None, delay=0.2, max_wait=0.3, colored=True):
    """
    Inputs:
        - asp: an adversarial search problem
        - bots: a list in which the i'th element is the bot for player i
        - visualizer (optional): a void function that takes in a game state
          and does something to visualize it. If no visualizer argument is
          passed, run_game will not visualize games.

    Runs a game and outputs the evaluation of the terminal state.
    """
    state = asp.get_start_state()
    if not visualizer == None:
        visualizer(state, colored)
        time.sleep(delay)

    while not (asp.is_terminal_state(state)):
        exposed = copy.deepcopy(asp)
        signal.signal(signal.SIGALRM, support.timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, max_wait)
        try:
            # run AI
            decision = bots[state.ptm].decide(exposed)
        except support.TimeoutException as msg:
            if visualizer:
                print(
                    """Warning. Player %s took too long to decide on a move.
They will go UP this round."""
                    % (state.ptm + 1)
                )
            decision = "U"
        signal.setitimer(signal.ITIMER_REAL, 0)

        available_actions = asp.get_available_actions(state)
        if not decision in available_actions:
            decision = list(available_actions)[0]

        result_state = asp.transition(state, decision)
        asp.set_start_state(result_state)

        state = result_state
        if not visualizer == None:
            visualizer(state, colored)
            time.sleep(delay)

    return asp.evaluate_state(asp.get_start_state())


# Note to self: clean this up for students.


def main():
    random.seed(1)
    np.random.seed(1)

    parser = argparse.ArgumentParser(prog="gamerunner", usage="%(prog)s [options]")
    parser.add_argument(
        "-map", type=str, help=HelpMessage.MAP, default=Argument_Defaults.MAP
    )
    parser.add_argument(
        "-max_wait",
        type=float,
        help=HelpMessage.MAX_WAIT,
        default=Argument_Defaults.MAX_WAIT,
    )
    parser.add_argument(
        "-bots",
        type=str,
        nargs="+",
        help=HelpMessage.BOTS,
        default=Argument_Defaults.BOTS,
    )
    parser.add_argument(
        "-image_delay",
        type=float,
        help=HelpMessage.IMAGE_DELAY,
        default=Argument_Defaults.IMAGE_DELAY,
    )
    parser.add_argument(
        "-no_image", dest="show_image", help=HelpMessage.NO_IMAGE, action="store_false"
    )
    parser.set_defaults(show_image=True)
    parser.add_argument("-multi_test", type=int, help=HelpMessage.MULTI_TEST)
    parser.add_argument(
        "-no_color", dest="colored", help=HelpMessage.COLOR, action="store_false"
    )
    parser.set_defaults(colored=True)

    args = parser.parse_args()

    wait = args.max_wait
    bots = support.determine_bot_functions(args.bots)
    delay = args.image_delay
    verbose = args.show_image
    multi = args.multi_test
    colored = args.colored

    game = TronProblem(args.map, 0)

    visualizer = None
    if verbose:
        visualizer = TronProblem.visualize_state

    if multi is not None:
        winners = defaultdict(int)
        bots = support.determine_bot_functions(args.bots)
        for i in range(multi):
            outcome = run_game(
                copy.deepcopy(game), bots, visualizer, delay, wait, colored
            )
            winner = outcome.index(1)
            winners[winner] += 1
            for bot in bots:
                bot.cleanup()
        for winner, wins in list(winners.items()):
            print("Player %s won %d out of %d times" % (winner + 1, wins, multi))

    else:
        outcome = run_game(copy.deepcopy(game), bots, visualizer, delay, wait, colored)
        winner = outcome.index(1) + 1
        print("Player %s won!" % winner)


class Argument_Defaults:
    MAP = "./maps/empty_room.txt"
    MAX_WAIT = 0.3
    BOTS = ["random", "random"]
    IMAGE_DELAY = 0.2


class HelpMessage:
    MAP = (
        '''the filename of the map to use for this game.
        Defaults to "'''
        + Argument_Defaults.MAP
        + """."""
    )
    MAX_WAIT = (
        """The amount of time (in seconds) the game engine will wait
        for a player to decide what move they want to make. If the player takes too long,
        they go north. Defaults to """
        + str(Argument_Defaults.MAX_WAIT)
        + """ (this
        will be reset during grading)."""
    )
    BOTS = (
        '''which bot each player will use. Valid bot types include "student",
        "wall", "random", "ta1", "ta2". This argument takes in a sequnce of bot types,
        where the first bot is used for the first player, the second bot is for the second
        player, and so on. Defaults to "'''
        + Argument_Defaults.BOTS[0]
        + """
         """
        + Argument_Defaults.BOTS[1]
        + """". Note that errors will occur if there
        are not enough AIs for the number of players on the board."""
    )
    IMAGE_DELAY = (
        """The amount of time (in seconds) to wait after printing the current
        state of the game. This is just to give users more time to watch the game progress.
        Defaults to """
        + str(Argument_Defaults.IMAGE_DELAY)
        + """."""
    )
    NO_IMAGE = """include this flag (with no arguments) to suppress the output of all
        board states and just get final results"""
    MULTI_TEST = """Test this map (multi_test) times in a row. Useful if you want to see how
        randomized algorithms do on average. It's recommended that you turn off verbose for this.
        Tracks how many times each player won across all games."""
    COLOR = """include this flag to remove the coloration in the output of board states. Use this if
        the coloring does not display properly"""


if __name__ == "__main__":
    main()

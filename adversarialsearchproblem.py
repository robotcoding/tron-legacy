from abc import ABCMeta, abstractmethod


class GameState(metaclass=ABCMeta):
    @abstractmethod
    def player_to_move(self):
        """
		Produces the index of the player who will move next.
		"""
        pass


class AdversarialSearchProblem(metaclass=ABCMeta):
    def get_start_state(self):
        """
		Produces the state from which to start.
		"""
        return self._start_state

    def set_start_state(self, state):
        """
		Changes the start state to the given state.
		"""
        self._start_state = state

    @abstractmethod
    def get_available_actions(self, state):
        """
		Returns the set of actions available to the player-to-move
		from the current state.
		"""
        pass

    @abstractmethod
    def transition(self, state, action):
        """
		Returns the state that results from taking the given action
		from the given state. (Assume deterministic transitions.)
		"""
        assert not (self.is_terminal_state(state))
        assert action in self.get_available_actions(state)
        pass

    @abstractmethod
    def is_terminal_state(self, state):
        """
		Produces a boolean indicating whether or not the given state is terminal.
		"""
        pass

    @abstractmethod
    def evaluate_state(self, state):
        """
		Takes in a terminal state, and, if the state is terminal,
		produces a list of nonnegative numbers that sum to 1, where the i'th
		element of the list is the value of the state to player i.
		Most commonly, this list will have a 1 at some index j, and all 0's
		everywhere else, indicating that player j won the game.
		"""
        assert self.is_terminal_state(state)
        pass

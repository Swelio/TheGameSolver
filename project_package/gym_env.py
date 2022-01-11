from typing import List

import gym
import numpy as np
from gym import spaces
from the_game.game_env import GameEnv

from project_package.learning_player import InvalidActionError, LearningPlayer


class GymGameEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.game_env: GameEnv = None
        self.player: LearningPlayer = None
        self.action_space = spaces.Discrete(len(LearningPlayer.ACTIONS))
        self.observation_space: spaces.Box = spaces.Box(
            0, 100, shape=(len(LearningPlayer.OBSERVATION),), dtype=np.int32
        )

    def step(self, action) -> (List[int], float, bool, dict):
        """
        Play action, as playing card on heap or end player turn (then draw).
        :param action: index of action to do.
        :type action: int
        :return: observation, reward, done, info
        :rtype: List[int], float, bool, dict
        """

        def _get_result(score: float, done: bool):
            return (
                self.player.observation,
                score,
                done,
                {
                    # "hand": self.player.hand,
                    # "heaps": self.player.heaps,
                    # "remaining": self.game_env.remaining_cards,
                },
            )

        try:
            player_gain = self.player.play_action(action)
        except InvalidActionError:
            return _get_result(0, False)

        # draw
        if self.game_env.remaining_cards > 0 and (
            self.player.ACTIONS[action] == self.player.PASS_ACTION
            or len(self.player.hand) == 0
        ):
            self.player.fill_hand()

        return _get_result(player_gain, self.player.can_play() is False)

    def reset(self):
        """
        Init game env and add player to it.
        Then fill player hand and return observation.
        :return:
        :rtype:
        """
        self.game_env = GameEnv()
        self.player: LearningPlayer = self.game_env.add_player(LearningPlayer)  # noqa
        self.game_env.prepare_game()
        return self.player.observation

    def render(self, mode="human"):
        if mode == "human":
            return self.player.observation

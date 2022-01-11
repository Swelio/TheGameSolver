from typing import Dict, List

import numpy as np
from tensorflow import keras
from the_game.game_env import GameEnv, Heap, Player


class InvalidActionError(Exception):
    """
    Raise on invalid action from LearningPlayer.
    Useful for learning.
    """


class LearningPlayer(Player):
    CARD_ACTIONS = [
        f"card_{x}::{action}"
        for x in range(8)
        for action in [
            "heap_up_1",
            "heap_up_2",
            "heap_down_1",
            "heap_down_2",
        ]
    ]

    PASS_ACTION = "pass"
    ACTIONS = [
        *CARD_ACTIONS,
        PASS_ACTION,
    ]

    OBSERVATION = [
        "heap_up_1",
        "heap_up_2",
        "heap_down_1",
        "heap_down_2",
        *[f"card_{x}" for x in range(8)],
    ]

    def __init__(self, game_env: GameEnv):
        super().__init__(game_env)
        self.turn_played_cards = 0
        self.model: keras.Model = None

    @property
    def heaps(self) -> Dict[str, Heap]:
        return {
            "heap_up_1": self.game_env.heaps[Heap.HEAP_UP][0],
            "heap_up_2": self.game_env.heaps[Heap.HEAP_UP][1],
            "heap_down_1": self.game_env.heaps[Heap.HEAP_DOWN][0],
            "heap_down_2": self.game_env.heaps[Heap.HEAP_DOWN][1],
        }

    @property
    def observation(self) -> List[int]:
        """
        Return observation for neural network.
        :return: [
                    heap_up_1,
                    heap_up_2,
                    heap_down_1,
                    heap_down_2,
                    hand_cards...
                    ]
        :rtype: List[int]
        """

        observation = []

        for heap in (
            self.game_env.heaps[Heap.HEAP_UP] + self.game_env.heaps[Heap.HEAP_DOWN]
        ):
            observation.append(heap.heap[0])

        observation.extend(sorted(self.hand))
        observation.extend([0] * (8 - len(self.hand)))

        return observation

    def play_action(self, action) -> int:
        """
        Play action according to its index.
        Return the gain on heap.
        :param action:
        :type action:
        :return:
        :rtype:
        """
        if self.ACTIONS[action] == self.PASS_ACTION:
            if (self.game_env.remaining_cards > 0 and self.turn_played_cards < 2) or (
                self.game_env.remaining_cards == self.turn_played_cards == 0
            ):
                raise InvalidActionError

            self.turn_played_cards = 0
            return 50

        sorted_hand = sorted(self.hand)
        card_label, heap_label = self.ACTIONS[action].split("::")
        card_index = int(card_label.split("_")[1])

        # invalid selection from neural network
        if card_index >= len(sorted_hand):
            raise InvalidActionError

        played_heap = self.heaps[heap_label]
        played_card = sorted_hand[card_index]

        # compute gain
        if played_heap.direction == Heap.HEAP_UP:
            gain = played_heap.heap[0] - played_card
        else:
            gain = played_card - played_heap.heap[0]

        if self.play_card(played_card, played_heap) is True:
            self.turn_played_cards += 1
            return gain + 100
        else:
            raise InvalidActionError

    def can_play(self) -> bool:
        """
        Check if player can play with his current hand.
        :return:
        :rtype:
        """
        for card in self.hand:
            for heap in self.game_env.heap_list:
                if heap.validate_card(card) is True:
                    return True
        return False

    def play(self):
        assert self.model is not None, "Please setup model."

        while self.can_play():

            observation = np.array(self.observation)
            predictions = self.model.predict(np.expand_dims(observation, axis=0))[0]

            for _ in range(len(predictions)):
                selected_action: int = np.argmax(predictions)

                try:
                    result = self.play_action(selected_action)
                except InvalidActionError:
                    predictions[selected_action] = 0.0
                    continue

                print("Observation:", observation)
                print("Played:", LearningPlayer.ACTIONS[selected_action])

                if result == 50:
                    print("Draw.")
                    return
                break

import os
import sys

from tensorflow import keras
from the_game import GameEnv

from project_package.learning_player import LearningPlayer
from training import MODEL_PATH, WEIGHTS_PATH

if __name__ == "__main__":

    assert os.path.exists(MODEL_PATH) and os.path.isfile(MODEL_PATH)
    assert os.path.exists(WEIGHTS_PATH) and os.path.isfile(WEIGHTS_PATH)

    with open("model_structure.json", "r") as model_file:
        json_model = model_file.read()
    model = keras.models.model_from_json(json_model)

    try:
        model.load_weights(WEIGHTS_PATH)
    except ValueError:
        print("Invalid weights file.")
        sys.exit(1)

    game_env = GameEnv()
    player = game_env.add_player(LearningPlayer)
    player.model = model
    game_env.play_game()
    print("Remaining cards:", game_env.remaining_cards)
    print("Played cards:", game_env.played_cards)

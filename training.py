# Source: https://keras.io/examples/rl/deep_q_network_breakout/

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from project_package.gym_env import GymGameEnv

MODEL_PATH = "model_structure.json"
WEIGHTS_PATH = "TheGamePlayer.h5"

NEEDED_SCORE = 1100
EPISODE_REWARD_LENGTH = 500

seed = 42
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min

batch_size = 32
max_steps_per_episode = 10000


def create_q_model():
    input_layer = layers.Input(shape=game_env.observation_space.shape)
    layer = layers.Dense(1024, activation="relu")(input_layer)
    layer = layers.Dropout(0.1)(layer)
    layer = layers.Dense(2048, activation="relu")(layer)
    layer = layers.Dropout(0.1)(layer)
    layer = layers.Dense(1024, activation="relu")(layer)
    layer = layers.Dropout(0.1)(layer)
    output_layer = layers.Dense(game_env.action_space.n, activation="linear")(layer)

    return keras.Model(inputs=input_layer, outputs=output_layer)


if __name__ == "__main__":
    game_env = GymGameEnv()

    if os.path.exists(MODEL_PATH) and os.path.isfile(MODEL_PATH):
        with open("model_structure.json", "r") as model_file:
            json_model = model_file.read()
        model = keras.models.model_from_json(json_model)
        model_target = keras.models.model_from_json(json_model)

    else:
        model = create_q_model()
        model_target = create_q_model()

    if os.path.exists(WEIGHTS_PATH) and os.path.isfile(WEIGHTS_PATH):
        try:
            model.load_weights(WEIGHTS_PATH)
            model_target.load_weights(WEIGHTS_PATH)
        except ValueError:
            print("Invalid weights file")

    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    action_history = []

    state_history = []
    state_next_history = []

    rewards_history = []
    episode_reward_history = []

    done_history = []

    running_reward = 0
    episode_count = 0
    frame_count = 0

    epsilon_random_frames = 50000
    epsilon_greedy_frames = 1000000.0
    max_memory_length = 100000

    update_after_actions = 4
    update_target_network = 10000

    loss_function = keras.losses.Huber()

    while running_reward < NEEDED_SCORE:
        state = np.array(game_env.reset())
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            frame_count += 1

            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                action = np.random.randint(0, game_env.action_space.n)
            else:
                state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
                action_probs = model(state_tensor, training=False)[0]
                action = tf.argmax(action_probs).numpy()

            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            state_next, reward, done, _ = game_env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)

            state = state_next

            if (
                frame_count % update_after_actions == 0
                and len(done_history) > batch_size
            ):
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )
                future_rewards = model_target.predict(state_next_sample)

                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                masks = tf.one_hot(action_sample, game_env.action_space.n)

                with tf.GradientTape() as tape:
                    q_values = model(state_sample)
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    loss = loss_function(updated_q_values, q_action)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % update_target_network == 0:
                model_target.set_weights(model.get_weights())
                if episode_count >= EPISODE_REWARD_LENGTH:
                    print(
                        f"running reward: {running_reward:.2f} "
                        f"at episode {episode_count}, "
                        f"frame count {frame_count}"
                    )

            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > EPISODE_REWARD_LENGTH:
            del episode_reward_history[:1]

        if len(episode_reward_history) == EPISODE_REWARD_LENGTH:
            running_reward = np.mean(episode_reward_history)
        episode_count += 1

    print(f"Solved at episode {episode_count} with running reward {running_reward}.")
    model_target.save_weights("TheGamePlayer.h5", overwrite=True)

    if not os.path.exists("model_structure.json"):
        with open("model_structure.json", "w") as model_file:
            model_file.write(model_target.to_json())

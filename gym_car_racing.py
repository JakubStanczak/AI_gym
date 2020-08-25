import gym
import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
from collections import Counter

import matplotlib.pyplot as plt

env = gym.make('CarRacing-v0')

POSSIBLE_ACTIONS = 4
EPISODES = 1 + 1
EPISODE_MAX_LEN = 1_000
REPLAY_MEMORY_SIZE = 20_000
RANDOM_MOVES = 1_000

WILL_TO_EXPLORE = 0.9
WILL_MIN = 0.2

will_decay_step = WILL_TO_EXPLORE / (EPISODES - RANDOM_MOVES)

TRAIN_EVERY = 100
TRAIN_BATCH = 5_000
Q_DISCOUNT = 0.95


observation_size = env.observation_space.shape[0] * env.observation_space.shape[1]
print(observation_size)
# print(env.observation_space.high)
# print(env.observation_space.low)
print(env.action_space.high)
print(env.action_space.low)


class Agent:
    def get_nn(self):
        model = Sequential([
            Dense(256, input_dim=observation_size, activation='relu'),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(POSSIBLE_ACTIONS)
        ])

        model.compile(loss='mse', optimizer=Adam(lr=0.0005))
        return model

    def __init__(self):
        self.train_model = self.get_nn()
        self.pred_model = self.get_nn()
        self.pred_model.set_weights(self.train_model.get_weights())
        self.train_counter = 0
        # observation, reward, done, action, new_q
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # action, reward, done
        self.score_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.will_to_explore = WILL_TO_EXPLORE

    def take_action(self, observation):
        action, q = self.chose_action(observation)
        new_observation, reward, done, _ = env.step(action)


        return new_observation, done

    def chose_action(self, observation):
        # print(observation.shape)
        observation = observation[:, :, 1] / 255
        # print(observation.shape)
        observation = np.reshape(observation, (1, -1))
        # print(observation.shape)
        # print(np.max(observation))
        # print(np.min(observation))
        q = self.pred_model.predict(observation)
        if len(self.replay_memory) < RANDOM_MOVES or np.random.random() < self.will_to_explore:
            action = env.action_space.sample()
        else:
            print(q)
            action_n = np.argmax(q)
            if action_n == 0:
                action = [1, 0, 0]
            elif action_n == 1:
                action = [-1, 0, 0]
            elif action_n == 2:
                action = [0, 1, 0]
            elif action_n == 3:
                action = [0, 0, 1]
            else:
                print('ERROR: action_n too high')
                action = env.action_space.sample()
        return action, q

    def will_decay(self):
        if len(self.replay_memory) < RANDOM_MOVES and self.will_to_explore - will_decay_step > WILL_MIN:
            self.will_to_explore -= will_decay_step


agent = Agent()

for episode in range(EPISODES):
    print('episode', episode, 'will_to_explore', agent.will_to_explore)
    observation = env.reset()
    move = 0
    done = False
    # gathered_reward = 0
    # max_reward = 0
    # min_reward = 0
    while not done:
        move += 1
        new_observation, done = agent.take_action(observation)
        observation = new_observation
        env.render()

        if move >= EPISODE_MAX_LEN:
            done = True

env.close()

def manual_play():
    env.reset()
    game_lost = False
    while not game_lost:
        env.render()

        key_not_chosen = True
        while key_not_chosen:
            action = input('input a for left d for right')
            if action == 'd':
                action = [1, 0, 0]
                key_not_chosen = False
            elif action == 'a':
                action = [-1, 0, 0]
                key_not_chosen = False
            elif action == 'w':
                action = [0, 1, 0]
                key_not_chosen = False
            # else:
            #     action = 0
            #     key_not_chosen = False
        observation, reward, done, info = env.step(action)
        print('observation, reward, done, info')
        print(observation, reward, done, info)
        if done:
            print('YOU LOST')
            game_lost = True
    env.close()

# manual_play()

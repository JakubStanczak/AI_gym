import gym
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

pd.set_option('display.max_columns', 250)

env = gym.make('CartPole-v0')

def manual_play():
    env.reset()
    game_lost = False
    while not game_lost:
        env.render()

        key_not_chosen = True
        while key_not_chosen:
            action = input('input a for left d for right')
            if action == 'a':
                action = 0
                key_not_chosen = False
            elif action == 'd':
                action = 1
                key_not_chosen = False
        observation, reward, done, info = env.step(action)
        print('observation, reward, done, info')
        print(observation, reward, done, info)
        if done:
            print('YOU LOST')
            game_lost = True
    env.close()

def save_move(game_num, prev_observation, action, good_action):
    saved_moves.loc[len(saved_moves)] = [game_num] + list(prev_observation) + [action] + [good_action]


def generate_random_moves():
    scores = []
    for game_num in range(5):
        observation = env.reset()
        for move_num in range(100): # max number of moves
            env.render()
            action = env.action_space.sample()
            prev_observation = observation
            observation, reward, done, info = env.step(action)
            saved_moves.loc[len(saved_moves)] = [game_num] + list(prev_observation) + [action] + [not bool(done)]
            if done:
                scores.append(move_num)
                break
    env.close()
    return scores

def test_model(model):
    for game_num in range(5):
        observation = env.reset()
        for move_num in range(1000): # max number of moves
            env.render()

            observation = np.reshape(observation, (1, -1))
            action = model.predict(observation)
            action = np.argmax(action)

            observation, reward, done, info = env.step(action)
            if done:
                print('This model lasted {} moves'.format(move_num))
                break
    env.close()



def run_template():
    for i_episode in range(2):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            print(action)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

observation = env.reset()
print(observation.shape)

# initialize df
moves_df_architecture = {}
observation_size = len(env.observation_space.high)
moves_df_architecture['game_num'] = []
for i in range(observation_size):
    moves_df_architecture['observation_before' + str(i)] = []
moves_df_architecture['action'] = []
moves_df_architecture['good_action'] = []

saved_moves = pd.DataFrame(moves_df_architecture)


scores_0 = generate_random_moves()
print('Average score when random moves {}'.format(np.mean(scores_0)))

saved_moves = saved_moves[ saved_moves['good_action'] == 1 ]

x_columns = [x for x in saved_moves.columns if 'observation_before' in x]
X = saved_moves[x_columns].values
y = saved_moves[['action']].values
y = to_categorical(y)

test_num = 10

X_train = X[:-test_num]
X_test = X[-test_num:]
y_train = y[:-test_num]
y_test = y[-test_num:]

print('Array sizes')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


model = Sequential([
    Dense(512, input_dim=observation_size, activation='relu'),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


model.fit(X_train, y_train,
          batch_size=1024, epochs=20, verbose=2,
          validation_data=(X_test, y_test))


# y_pred = np.array([0.2, 0.3, 0.4, 0.5])
# print(type(y_pred))
# print(y_pred)
# print(y_pred.shape)
# y_pred = np.reshape(y_pred, (1,-1))
# print(y_pred)
# print(y_pred.shape)
#
# pred = model.predict(y_pred)
# print(pred)
# print(np.argmax(pred))
test_model(model)
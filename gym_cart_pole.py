import gym
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

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


def generate_random_moves():
    print('generating random starting data')
    for game_num in tqdm(range(500)):
        observation = env.reset()
        for move_num in range(100): # max number of moves
            # env.render()
            action = env.action_space.sample()
            prev_observation = observation
            observation, reward, done, info = env.step(action)
            saved_moves.loc[len(saved_moves)] = [game_num] + list(prev_observation) + [action] + [not bool(done)]
            if done:
                break
    env.close()


def test_model(model):
    freq_of_rand = 20
    for game_num in range(5):
        chosen_actions = []
        observation = env.reset()
        for move_num in range(1000):
            env.render()
            observation = np.reshape(observation, (1, -1))
            if move_num % freq_of_rand == 0:
                action = env.action_space.sample()
            else:
                action = model.predict(observation)
                action = np.argmax(action)
                chosen_actions.append(action)
            observation, reward, done, info = env.step(action)
            if done and move_num == 199:
                print('\nThis model lasted 200 frames with random moves every {} frames throwing it off balance'.format(freq_of_rand))
                print('It was too easy.Difficulty level increased!!')
                freq_of_rand -= 5
                break
            elif done:
                print('\nThis model lasted {} moves with random moves every {} frames throwing it off balance'.format(move_num, freq_of_rand))
                break
        print(Counter(chosen_actions))
    env.close()


observation_size = len(env.observation_space.high)
def define_df():
    moves_df_architecture = {}
    moves_df_architecture['game_num'] = []
    for i in range(observation_size):
        moves_df_architecture['observation_before' + str(i)] = []
    moves_df_architecture['action'] = []
    moves_df_architecture['good_action'] = []
    return moves_df_architecture


min_accepted_score = 40
def prune_df():
    global saved_moves
    # TODO delete 5 last moves before lost
    games_played = max(saved_moves['game_num'])
    scores = saved_moves.groupby(['game_num']).size()
    saved_moves['score'] = saved_moves['game_num'].map(scores)
    saved_moves = saved_moves[ saved_moves['score'] > min_accepted_score]
    saved_moves = saved_moves[saved_moves['good_action'] == 1]

    games_left = len(saved_moves['game_num'].unique())
    not_accepted_games = games_played - games_left
    print('average score was {} and the best score was {}'.format(np.mean(saved_moves['score']), np.max(saved_moves['score'])))
    print('{} games were not accepted \nthis leaves us with {} games and {} records'.format(not_accepted_games, games_left, saved_moves.shape[0]))



# initialize df
df_architecture = define_df()
saved_moves = pd.DataFrame(df_architecture)

generate_random_moves()
prune_df()

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
    Dense(1024, input_dim=observation_size, activation='relu'),
    Dropout(0.2),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


model.fit(X_train, y_train,
          batch_size=128, epochs=5, verbose=2,
          validation_data=(X_test, y_test))

test_model(model)
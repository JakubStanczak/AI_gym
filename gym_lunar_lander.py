import gym
import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
from collections import Counter

import matplotlib.pyplot as plt


env = gym.make('LunarLander-v2')

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)
observation_size = len(env.observation_space.high)


# !!!!!!!!!!!!!!!!!! Now model? than iteration 0 model_dir None
model_iteration = 0
# model_dir = 'model_weights/model_0_iteration_500_episode'
model_dir = None
only_replay = False


# initializing model
model = Sequential([
    Dense(512, input_dim=observation_size, activation='relu'),
    Dropout(0.2),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(env.action_space.n)
])

model.compile(loss='mse', optimizer=Adam(lr=0.001))
# print(model.summary())

if model_dir is not None:
    model.load_weights(model_dir)

def replay():
    observation = env.reset()
    env.render()
    done = False
    agg_reward = 0
    while not done:
        observation = np.reshape(observation, (1, -1))
        q = model.predict(observation)
        action = np.argmax(q)
        # print('q', q)
        # print('action', action)
        observation, reward, done, _ = env.step(action)
        agg_reward += reward

        env.render()
    print('reward', agg_reward)

if only_replay:
    while True:
        replay()


episodes = 2_000
discount = 0.95
render_every_x_episodes = 100

max_last_actions = 50_000
min_last_actions = 5_000
use_last_actions = 3_000
observation_memory = deque(maxlen=max_last_actions)
new_q_memory = deque(maxlen=max_last_actions)
not_crashed_memory = deque(maxlen=max_last_actions)
last_x_scores = deque(maxlen=50)

will_to_explore = 0.99
will_decay_start = episodes // 10
will_decay_stop = episodes
will_min = 0.3

will_decay_step = will_to_explore / (will_decay_stop - will_decay_start)

episode_plot = []
min_reward_plot = []
max_reward_plot = []
gathered_reward_plot = []
average_reward_per_x_plot = []

actions_to_solve = []
for episode in range(episodes):
    print('episode', episode, 'will_to_explore', will_to_explore)
    observation = env.reset()
    done = False
    gathered_reward = 0
    max_reward = 0
    min_reward = 0
    if episode >= will_decay_start and will_to_explore - will_decay_step > will_min:
        will_to_explore -= will_decay_step
    while not done:
        observation = np.reshape(observation, (1, -1))
        observation_memory.append(observation)
        q = model.predict(observation)
        if np.random.random() < will_to_explore:
            action = env.action_space.sample()
        else:
            action = np.argmax(q)
        new_observation, reward, done, _ = env.step(action)
        if reward > max_reward: max_reward = reward
        if reward < min_reward: min_reward = reward
        gathered_reward += reward
        new_observation = np.reshape(new_observation, (1, -1))
        next_q = model.predict(new_observation)
        if not done:
            new_q_value = reward + discount * np.max(next_q)
        else:
            new_q_value = reward
        # print('q', q)
        # print('new_q_value', new_q_value)
        q[0, np.argmax(q)] = new_q_value
        new_q_memory.append(q)
        # print('new_q', q)

        observation = new_observation
        # if done and reward > 100:
        #     print('Tasked solved in episode {} with {} points'.format(episode, reward))
        #     actions_to_solve = actions_taken
        if done and len(observation_memory) >= min_last_actions:
            idxs = random.choices(range(len(observation_memory)), k=use_last_actions)
            X = np.array(observation_memory[0])
            y = np.array(new_q_memory[0])
            for idx in idxs:
                X = np.vstack((X, observation_memory[idx]))
                y = np.vstack((y, new_q_memory[idx]))

            # print('X', X)
            # print('y', y)
            # print('X shape', X.shape)
            # print('y shape', y.shape)
            model.fit(X, y, batch_size=32, epochs=2, verbose=0)


        if episode % render_every_x_episodes == 0 and episode > will_decay_start:
            model.save_weights('model_weights/model_{}_iteration_{}_episode'.format(model_iteration, episode))
            env.render()

    last_x_scores.append(gathered_reward)
    print('gathered_reward', gathered_reward, 'max_reward', max_reward, 'min_reward', min_reward)
    print('last_{}_scores mean'. format(len(last_x_scores)), np.mean(last_x_scores), 'observation_memory_len', len(observation_memory))
    print()

    episode_plot.append(episode)
    gathered_reward_plot.append(gathered_reward)
    min_reward_plot.append(min_reward)
    max_reward_plot.append(max_reward)
    average_reward_per_x_plot.append(np.mean(last_x_scores))

env.close()

plt.plot(episode_plot, gathered_reward_plot)
plt.plot(episode_plot, min_reward_plot)
plt.plot(episode_plot, max_reward_plot)
plt.plot(episode_plot, average_reward_per_x_plot)
plt.savefig('learning_plot.png')
plt.show()


while True:
    replay()
env.close()


def manual_play():
    env.reset()
    game_lost = False
    while not game_lost:
        env.render()

        key_not_chosen = True
        while key_not_chosen:
            action = input('input a for left d for right')
            if action == 'a':
                action = 1
                key_not_chosen = False
            elif action == 'd':
                action = 3
                key_not_chosen = False
            elif action == 'w':
                action = 2
                key_not_chosen = False
            else:
                action = 0
                key_not_chosen = False
        observation, reward, done, info = env.step(action)
        print('observation, reward, done, info')
        print(observation, reward, done, info)
        if done:
            print('YOU LOST')
            game_lost = True
    env.close()

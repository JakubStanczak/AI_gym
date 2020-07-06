import gym
import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from collections import deque

env = gym.make('LunarLander-v2')

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)
observation_size = len(env.observation_space.high)

# initializing model
model = Sequential([
    Dense(512, input_dim=observation_size, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(env.action_space.n, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

discount = 0.5
episodes = 100
render_every_x_episodes = 10

max_last_actions = 50_000
use_last_actions = 25_000
observation_memory = deque(maxlen=max_last_actions)
new_q_memory = deque(maxlen=max_last_actions)

will_to_explore = 0.9
will_decay_start = episodes // 3
will_decay_stop = episodes
will_decay_step = will_to_explore / (will_decay_stop - will_decay_start)

actions_to_solve = []
for episode in range(episodes):
    print('episode', episode)
    observation = env.reset()
    done = False
    actions_taken = []
    # new_qs = []
    gathered_reward = 0
    max_reward = 0
    min_reward = 0
    if episode >= will_decay_start and will_to_explore > 0:
        will_to_explore -= will_decay_step
    while not done:
        observation = np.reshape(observation, (1, -1))
        observation_memory.append(observation)
        q = model.predict(observation)
        if np.random.random() < will_to_explore:
            action = env.action_space.sample()
        else:
            action = np.argmax(q)
        actions_taken.append(action)
        new_observation, reward, done, _ = env.step(action)
        if reward > max_reward: max_reward = reward
        if reward < min_reward: min_reward = reward
        gathered_reward += reward
        new_observation = np.reshape(new_observation, (1, -1))
        next_q = model.predict(new_observation)
        reward /= 200
        if not done:
            new_q_value = reward + discount * np.max(next_q)
        else:
            new_q_value = reward * 2
        # print('new_q_value', new_q_value)
        # print('q', q)
        q[0, np.argmax(q)] = new_q_value
        new_q_memory.append(q)
        # print('new_q', q)

        observation = new_observation
        if done and reward > 100:
            print('Tasked solved in episode {} with {} points'.format(episode, reward))
            actions_to_solve = actions_taken
        elif done:
            idxs = random.choices(range(len(observation_memory)), k=use_last_actions)
            X = np.array(observation_memory[0])
            y = np.array(new_q_memory[0])
            for idx in idxs:
                X = np.vstack((X, observation_memory[idx]))
                y = np.vstack((y, new_q_memory[idx]))

            # print('X shape', X.shape)
            # print('y shape', y.shape)
            model.fit(X, y,
                      batch_size=1024, epochs=2, verbose=0)

        if episode % render_every_x_episodes == 0 and episode > will_decay_start:
            env.render()
    print('gathered_reward', gathered_reward, 'max_reward', max_reward, 'min_reward', min_reward)

env.close()

def replay():
    observation = env.reset()
    env.render()
    done = False
    while not done:
        observation = np.reshape(observation, (1, -1))
        q = model.predict(observation)
        action = np.argmax(q)
        observation, reward, done, _ = env.step(action)
        env.render()

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

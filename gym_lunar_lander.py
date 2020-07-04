import gym
import numpy as np

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

discount = 0.95
episodes = 1000
render_every_x_episodes = 10

# max_last_actions = 10000
# last_actions_used = 1000

actions_to_solve = []
for episode in range(episodes):
    print('episode', episode)
    observation = env.reset()
    done = False
    actions_taken = []
    while not done:
        observation = np.reshape(observation, (1, -1))
        q = model.predict(observation)
        action = np.argmax(q)
        actions_taken.append(action)
        new_observation, reward, done, _ = env.step(action)
        new_observation = np.reshape(new_observation, (1, -1))
        next_q = model.predict(new_observation)
        new_q_value = reward + discount * np.max(next_q)
        # print('q', q)
        q[0, np.argmax(q)] = new_q_value
        # print('new_q', q)
        model.fit(observation, q,
                  batch_size=1, epochs=2, verbose=0)

        observation=new_observation
        if done and reward > 100:
            print('Tasked solved in episode {} with {} points'.format(episode, reward))
            actions_to_solve = actions_taken

        if episode % render_every_x_episodes == 0:
            env.render()

env.close()

def replay_run(actions_taken):
    env.reset()
    env.render()
    for action_taken in actions_taken:
        env.step(action_taken)
        env.render()

while True:
    replay_run(actions_to_solve)
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

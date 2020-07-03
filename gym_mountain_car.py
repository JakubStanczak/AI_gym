import gym
import numpy as np

env = gym.make('MountainCar-v0')

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space)

learning_rate = 0.1
discount = 0.95
episodes = 1000

# preparing Q table
table_size = [20] * len(env.observation_space.high)
table_bucket_size = (env.observation_space.high - env.observation_space.low) / table_size
print('bucket_size', table_bucket_size)

q_table = np.random.uniform(low=-2, high=0, size=(table_size + [env.action_space.n]))
print(q_table.shape)

def observation_to_bean_num(observation):
    observation_bins = (observation - env.observation_space.low) // table_bucket_size
    return tuple(observation_bins.astype(np.int))

print('!!!')
observation = env.reset()
observation_bins = observation_to_bean_num(observation)
print(observation_bins)
print(q_table[observation_bins])

print(q_table[(1,2) ])
print((1, 2) + (3,))

for episode in range(episodes):
    observation = env.reset()
    observation_bins = observation_to_bean_num(observation)
    done = False
    while not done:
        action = np.argmax(q_table[observation_bins])
        observation, reward, done, _ = env.step(action)
        if observation[0] >= env.goal_position: # reward it this env does not work properly
            reward = 0
        else:
            reward = -1
        new_observation_bins = observation_to_bean_num(observation)
        env.render()

        max_next_q = np.max(q_table[new_observation_bins])
        used_q = q_table[observation_bins + (action, )]
        new_q = (1-learning_rate) * used_q + learning_rate * (reward + discount * max_next_q)
        q_table[observation_bins + (action, )] = new_q

        observation_bins = new_observation_bins

env.close()







def manual_play():
    env.reset()
    done = False
    while not done:
        env.render()
        key_not_chosen = True
        while key_not_chosen:
            action = input('input a for left d for right')
            if action == 'a':
                action = 0
                key_not_chosen = False
            elif action == 'd':
                action = 2
                key_not_chosen = False
        observation, reward, done, info = env.step(action)
        print('observation, reward, done, info')
        print(observation, reward, done, info)
        if done:
            env.render()
            print('YOU LOST')
    env.close()

manual_play()
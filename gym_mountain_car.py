import gym
import numpy as np

env = gym.make('MountainCar-v0')

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space)

learning_rate = 0.1
discount = 0.95
episodes = 3000
render_every_x_episodes = 100

# preparing Q table
table_size = [20] * len(env.observation_space.high)
table_bucket_size = (env.observation_space.high - env.observation_space.low) / table_size
# print('bucket_size', table_bucket_size)

q_table = np.random.uniform(low=-2, high=0, size=(table_size + [env.action_space.n]))
# print(q_table.shape)

will_to_explore = 0.5
will_decay_start = 500
will_decay_stop = 2000
will_decay_step = will_to_explore / (will_decay_stop - will_decay_start)

def observation_to_bean_num(observation):
    observation_bins = (observation - env.observation_space.low) // table_bucket_size
    return tuple(observation_bins.astype(np.int))

first_win = {'episode': None, 'action_num': None}
shortest_run = {'episode': 0, 'action_num': 200}
best_actions = []
for episode in range(episodes):
    observation = env.reset()
    observation_bins = observation_to_bean_num(observation)
    done = False
    action_num = 0
    actions_taken = []
    if episode >= will_decay_start and will_to_explore > 0:
        will_to_explore -= will_decay_step
    while not done:
        action_num += 1
        if np.random.random() < will_to_explore:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[observation_bins])
        actions_taken.append(action)
        observation, reward, done, _ = env.step(action)
        if observation[0] >= env.goal_position: # reward it this env does not work properly
            reward = 0
        else:
            reward = -1
        new_observation_bins = observation_to_bean_num(observation)

        if episode % render_every_x_episodes == 0:
            env.render()

        max_next_q = np.max(q_table[new_observation_bins])
        used_q = q_table[observation_bins + (action, )]
        new_q = (1-learning_rate) * used_q + learning_rate * (reward + discount * max_next_q)
        q_table[observation_bins + (action, )] = new_q
        observation_bins = new_observation_bins

        if done and action_num < 200:
            if first_win['episode'] is None:
                first_win['episode'] = episode
                first_win['action_num'] = action_num
                print('The task was solved by episode {} using {} moves'.format(episode, action_num))
            elif action_num < shortest_run['action_num']:
                shortest_run['episode'] = episode
                shortest_run['action_num'] = action_num
                best_actions = actions_taken
                print('shortest run from now on is {} from episode {}'.format(shortest_run['action_num'], shortest_run['episode']))

print('\nThe task was solved in {} episode using {} moves\nhowever the best run was in {} episode, it used {} moves\n'
      '\nNow the best run will be visualized'.format(first_win['episode'],
                                                     first_win['action_num'],
                                                     shortest_run['episode'],
                                                     shortest_run['action_num']))


def replay_run(actions_taken):
    env.reset()
    env.render()
    for action_taken in actions_taken:
        env.step(action_taken)
        env.render()

for _ in range(50):
    replay_run(best_actions)
env.close()

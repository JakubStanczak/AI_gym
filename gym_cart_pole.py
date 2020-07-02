import gym
import pandas as pd

env = gym.make('CartPole-v0')

print(env.action_space)
print(env.observation_space)

# initialize df
df = pd.DataFrame({'moves': [],
                   'observation_after_move': []})

def manual_play():
    observation = env.reset()
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

    pass


def run_template():
    print('sadfas')
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


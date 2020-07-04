import gym

env = gym.make('LunarLander-v2')

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space)



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

manual_play()
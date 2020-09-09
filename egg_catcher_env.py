import numpy as np
import random

import cv2 #bgr colors

class Basket:
    def __init__(self, env_size):
        self.x_size = 5
        self.y_size = 3
        self.x = env_size[1]//2 - self.x_size//2
        self.y = env_size[0]-1
        self.color = [255, 153, 102]

    def draw(self, env):
        for x in range(self.x, self.x+self.x_size+1):
            env[self.y, x] = self.color
        for y in range(self.y-self.y_size+1, self.y):
            env[y, self.x] = self.color
            env[y, self.x + self.x_size] = self.color

    def move(self, direction, env_size):
        if self.x + direction >= 0 and self.x + self.x_size + direction <= env_size[1] - 1:
            self.x += direction


class Egg:
    def __init__(self, env_size):
        self.size = 1
        self.speed = 1
        self.good = random.randint(0, 1)
        if self.good:
            self.color = [0, 255, 0]
        else:
            self.color = [0, 0, 255]
        self.x = random.randint(0, env_size[1]-1)
        self.y = 0

    def time(self, safe_zone_x, safe_zone_y, env_size):
        self.y += self.speed

        if self.good:
            prize_or_penalty = 1
        else:
            prize_or_penalty = -1

        # falling off screen
        if self.y == env_size[0]:
            return -10 * prize_or_penalty

        # caught with the basket
        if safe_zone_x[0] <= self.x <= safe_zone_x[1] and safe_zone_y[0] <= self.y <= safe_zone_y[1]:
            return 5 * prize_or_penalty

        # hitting the basket
        if (self.x == safe_zone_x[0] - 1 or self.x == safe_zone_x[1] + 1) and safe_zone_y[0] <= self.y <= safe_zone_y[1]:
            return -5 * prize_or_penalty

        return 0


    def draw(self, env):
        env[self.y, self.x] = self.color

    def __repr__(self):
        return f'(x {self.x}, y {self.y})'

class Egg_Catcher:
    def __init__(self, env_size=(30, 30, 3), render=True):
        self.size = env_size
        self.possible_actions = 3
        self.drop_every = 5
        self.env = np.zeros(self.size).astype(np.uint8)
        self.eggs = []
        self.add_egg(self.size)
        self.basket = Basket(self.size)
        self.move_count = 0
        self.max_moves = 500
        self.render = render
        if self.render:
            self.draw()

    def add_egg(self, env_size):
        self.eggs.append(Egg(env_size))

    def draw(self):
        env = np.zeros(self.size).astype(np.uint8)
        self.basket.draw(env)
        for egg in self.eggs:
            egg.draw(env)

        show_env = cv2.resize(env, (self.size[0] * 10, self.size[1] * 10), interpolation=cv2.INTER_AREA)
        cv2.imshow('image', show_env)
        cv2.waitKey(1)

    def time(self):
        del_eggs = []
        total_reward = 0
        for egg in self.eggs:
            reward = egg.time([self.basket.x + 1, self.basket.x + self.basket.x_size - 1], [self.basket.y - self.basket.y_size + 1, self.basket.y], self.size)
            if reward != 0:
                total_reward += reward
                del_eggs.append(egg)

        for egg in del_eggs:
            self.eggs.pop(self.eggs.index(egg))

        return total_reward

    def execute_move(self, chosen_move):
        self.move_count += 1

        if chosen_move == 0:
            direction = 0
        elif chosen_move == 1:
            direction = -1
        else:
            direction = 1

        self.basket.move(direction, self.size)
        reward = self.time()

        if self.move_count % self.drop_every == 0:
            self.add_egg(self.size)
        if self.render:
            self.draw()

        if self.move_count >= self.max_moves:
            done = True
            self.move_count = 0
        else:
            done = False
        return self.env, reward, done


manual_test = False

legal_moves = ['a', 'd', 's']
def move():
    move_legal = False
    while not move_legal:
        chosen_move = input('your move')

        if chosen_move in legal_moves:
            move_legal = True
        else:
            print('illegal move!!')

    if chosen_move == 'a':
        chosen_move = 1
    elif chosen_move == 'd':
        chosen_move = 2
    else:
        chosen_move = 0

    return chosen_move

if manual_test:
    env = Egg_Catcher()

    done = False
    while not done:
        chosen_move = move()
        observation, reward, done = env.execute_move(chosen_move)
        print(reward)


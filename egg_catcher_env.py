import gym
import numpy as np
import random

import cv2 #bgr colors
import threading

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from collections import deque
from collections import Counter

import matplotlib.pyplot as plt

ENV_SIZE = (30, 30, 3)
drop_every = 5



env = np.zeros(ENV_SIZE).astype(np.uint8)

class Basket:
    def __init__(self):
        self.x_size = 5
        self.y_size = 3
        self.x = ENV_SIZE[1]//2 - self.x_size//2
        self.y = ENV_SIZE[0]-1
        self.color = [255, 153, 102]

    def draw(self, env):
        for x in range(self.x, self.x+self.x_size+1):
            env[self.y, x] = self.color
        for y in range(self.y-self.y_size+1, self.y):
            env[y, self.x] = self.color
            env[y, self.x + self.x_size] = self.color

    def move(self, dir):
        if self.x + dir >=0 and self.x + dir < ENV_SIZE[1] - 1:
            self.x += dir

class Egg:
    def __init__(self):
        self.size = 1
        self.speed = 1
        self.good = True
        self.color = [0, 255, 0]
        self.x = random.randint(0, ENV_SIZE[1]-1)
        self.y = 0

    def time(self, safe_zone_x, safe_zone_y):
        self.y += 1

        if self.y == ENV_SIZE[0]:
            return -10
        if safe_zone_x[0] <= self.x <= safe_zone_x[1] and safe_zone_y[0] <= self.y <= safe_zone_y[1]:
            return 1

        if (self.x == safe_zone_x[0] - 1 or self.x == safe_zone_x[1] + 1) and safe_zone_y[0] <= self.y <= safe_zone_y[1]:
            return -10
        return 0


    def draw(self, env):
        env[self.y, self.x] = self.color

    def __repr__(self):
        return f'(x {self.x}, y {self.y})'

def add_egg():
    eggs.append(Egg())

def draw():
    env = np.zeros(ENV_SIZE).astype(np.uint8)

    basket.draw(env)
    for egg in eggs:
        egg.draw(env)


    show_env = cv2.resize(env, (ENV_SIZE[0]*10, ENV_SIZE[1]*10), interpolation=cv2.INTER_AREA)
    cv2.imshow('image', show_env)
    cv2.waitKey(1)

def time():
    del_eggs = []
    for egg in eggs:
        reward = egg.time([basket.x+1, basket.x + basket.x_size-1], [basket.y - basket.y_size+1, basket.y])
        if reward != 0:
            del_eggs.append(egg)
            print(f'reward: {reward}')

    for egg in del_eggs:
        eggs.pop(eggs.index(egg))

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
        chosen_move = -1
    elif chosen_move == 'd':
        chosen_move = 1
    else:
        chosen_move = 0

    return chosen_move





# t = threading.Timer(1, time)
# t.start()
# t.do_run = False
eggs = []
basket = Basket()

while True:
    add_egg()
    draw()
    chosen_move = move()
    basket.move(chosen_move)
    time()
    draw()


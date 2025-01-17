import gym
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from collections import deque
from collections import Counter

import matplotlib.pyplot as plt

import egg_catcher_env




from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)





EPISODES = 500 + 1
EPISODE_MAX_LEN = 500
DROP_EVERY = 10
DRAW_EVERY = 1


if DRAW_EVERY == 0:
	RENDER = False
else:
	RENDER = True

#save
SAVE_WEIGHTS_EVERY_EPISODES = 10
WEIGHT_SAVE_DIR = 'eggs_model_weights/model'
PLOT_SAVE_DIR = 'eggs_learning_plot'
PLOT_AVERAGE_REWARD_FOR_LAST = 20

# load
MODEL_ITERATION = 3
REPLAY = False
CONTINUE_TRAINING = False
LOAD_DIR = 'eggs_model_weights/model'


env = egg_catcher_env.Egg_Catcher(render=RENDER, max_moves=EPISODE_MAX_LEN, drop_every=DROP_EVERY)


observation_size = env.size
# print(env.observation_space.high)
# print(env.observation_space.low)

class Agent:
	def get_nn(self, cnn=True):
		if not cnn:
			model = Sequential([
				Dense(1024, input_dim=observation_size, activation='relu'),
				Dropout(0.2),
				Dense(512, activation='relu'),
				Dropout(0.2),
				Dense(256, activation='relu'),
				Dropout(0.2),
				Dense(self.possible_actions, activation='linear')
			])

		else:
			model = Sequential([
				Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=env.size),
				# MaxPool2D(pool_size=(2,2)),
				Dropout(0.2),

				Conv2D(512, kernel_size=(3, 3), activation='relu'),
				MaxPool2D(pool_size=(2,2)),
				Dropout(0.2),

				Flatten(),

				Dense(256, activation='relu'),
				Dense(self.possible_actions, activation='linear')
			])

		model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
		print(model.summary())
		return model

	def load_weights(self):
		self.pred_model = load_model(LOAD_DIR)
		self.train_model = load_model(LOAD_DIR)

	def __init__(self):
		self.possible_actions = env.possible_actions

		self.replay_memory_size = 20000
		self.random_moves = 1000
		self.start_will_to_explore = 0.9
		self.will_min = 0.2
		self.q_discount = 0.99
		self.train_batch_size = 64
		self.copy_weights_every_moves = 100

		self.will_decay_step = self.start_will_to_explore / EPISODES * 1.5
		self.train_model = self.get_nn()
		self.pred_model = self.get_nn()
		self.pred_model.set_weights(self.train_model.get_weights())
		if CONTINUE_TRAINING:
			self.load_weights()
		self.train_counter = 0
		# observation, reward, done, action, new_q
		self.replay_memory = deque(maxlen=self.replay_memory_size)
		self.will_to_explore = self.start_will_to_explore

	def reshape_observation(self, observation):
		# if observation.shape == (96, 96, 3):
		#     observation = observation[:, :, 1] / 255
		#     observation = np.reshape(observation, (1, -1))
		# print (f'shape before {observation.shape}')
		if observation.ndim < 4:
			observation = np.expand_dims(observation, axis=0)
			# print (f'shape after {observation.shape}')
			# print (f'max in observation {np.max(observation)}')
			observation = observation / 255
			# print (f'max in observation {np.max(observation)}')

		return observation

	def take_action(self, observation):
		observation = self.reshape_observation(observation)
		action, q = self.chose_action(observation)
		new_observation, reward, done = env.execute_move(action)
		new_observation = self.reshape_observation(new_observation)

		if not REPLAY:
			self.train(observation, new_observation, reward, done, q)
		return new_observation, reward, done

	def chose_action(self, observation):
		# print('observation shape', observation.shape)
		q = self.pred_model.predict(observation)
		action = np.argmax(q)

		if (len(self.replay_memory) < self.random_moves or np.random.random() < self.will_to_explore) and not REPLAY:
			# action = env.action_space.sample()
			action = random.randint(0, self.possible_actions - 1)

		return action, q

	def train(self, observation, new_observation, reward, done, q):
		next_q = self.pred_model.predict(new_observation)
		if not done:
			new_q_value = reward + self.q_discount * np.max(next_q)
		else:
			new_q_value = reward
		q[0, np.argmax(q)] = new_q_value

		self.replay_memory.append([observation, q])
		# print('moves in memory', len(self.replay_memory))

		if len(self.replay_memory) > self.train_batch_size * 2:
			X, y = self.pick_batch()
			self.train_model.fit(X, y, batch_size=self.train_batch_size, epochs=1, verbose=0)

		if self.train_counter >= self.copy_weights_every_moves:
			# print('copying weights')
			self.pred_model.set_weights(self.train_model.get_weights())
			self.train_counter = 0
		else:
			self.train_counter += 1

	def pick_batch(self):
		idxs = random.choices(range(len(self.replay_memory)), k=self.train_batch_size)
		X = np.array(self.replay_memory[0][0])
		y = np.array(self.replay_memory[0][1])
		for idx in idxs:
			X = np.vstack((X, self.replay_memory[idx][0]))
			y = np.vstack((y, self.replay_memory[idx][1]))
		return X, y

	def will_decay(self):
		if len(self.replay_memory) >= self.random_moves and self.will_to_explore - self.will_decay_step > self.will_min:
			self.will_to_explore -= self.will_decay_step


episode_plot = []
gathered_reward_plot = []
min_reward_plot = []
max_reward_plot = []
average_reward_for_last_plot = []
def update_score_plots(episode, reward, gathered_reward, min_reward, max_reward, done):
	if done:
		episode_plot.append(episode)
		gathered_reward_plot.append(gathered_reward)
		min_reward_plot.append(min_reward)
		max_reward_plot.append(max_reward)
		if len(average_reward_for_last_plot) > PLOT_AVERAGE_REWARD_FOR_LAST:
			average_reward_for_last_plot.append(np.mean(gathered_reward_plot[-PLOT_AVERAGE_REWARD_FOR_LAST:]))
		else:
			average_reward_for_last_plot.append(np.mean(gathered_reward_plot))
	else:
		if reward > max_reward:
			max_reward = reward
		if reward < min_reward:
			min_reward = reward
		gathered_reward += reward

	return min_reward, max_reward, gathered_reward

def learning_plot(episode, show=False):
	plt.plot(episode_plot, gathered_reward_plot, color='b')
	plt.plot(episode_plot, min_reward_plot, color='r')
	plt.plot(episode_plot, max_reward_plot, color='g')
	plt.plot(episode_plot, average_reward_for_last_plot, color='y')

	if show:
		plt.show()
	else:
		plt.savefig('{}/learning_plot_iteration_{}.png'.format(PLOT_SAVE_DIR, MODEL_ITERATION))


agent = Agent()

if not REPLAY:
	for episode in range(EPISODES):
		print('episode', episode, 'will_to_explore', agent.will_to_explore)
		observation = env.env
		move = 0
		done = False
		gathered_reward = 0
		max_reward = 0
		min_reward = 0
		agent.will_decay()
		while not done:
			move += 1
			new_observation, reward, done = agent.take_action(observation)
			# print(reward)
			observation = new_observation
			if RENDER:
				if episode % DRAW_EVERY == 0:
				    env.draw()

			if move >= EPISODE_MAX_LEN:
				done = True
			min_reward, max_reward, gathered_reward = update_score_plots(episode, reward, gathered_reward, min_reward, max_reward, done)
		print('moves', move, 'max_reward', max_reward, 'min_reward', min_reward, 'gathered_reward', gathered_reward)

		if episode % SAVE_WEIGHTS_EVERY_EPISODES == 0 and episode != 0:
			agent.train_model.save(WEIGHT_SAVE_DIR)
			learning_plot(episode, show=False)

else:
	while True:
		agent.load_weights()
		observation = env.env
		move = 0
		gathered_reward = 0
		done = False
		while not done:
			new_observation, reward, done = agent.take_action(observation)
			gathered_reward += reward
			observation = new_observation
			env.draw()
			
			move +=1
			if done:
				print('reward {} in {} moves'.format(gathered_reward, move))


learning_plot(None, show=True)


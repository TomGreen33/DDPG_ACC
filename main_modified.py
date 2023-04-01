# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:01:39 2023

@author: TOM3O
"""

CA = False
total_episode = 3000
TTC_threshold = 4.001


from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

import random
import numpy as np
from simulation_env_modified import Env
import scipy.io as sio
import pickle as pk
import sys
import os

from tqdm import tqdm_notebook as tqdm
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import aggregate_episodes
from tqdm import tqdm
import pandas as pd
import animate
import pygame
#%%
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# get_ipython().run_line_magic('matplotlib', 'inline')

#####################  hyper parameters  ####################
ACTOR_SIZE_A1 = 256
ACTOR_SIZE_A2 = 128

CRITIC_SIZE_A1 = 256
CRITIC_SIZE_A2 = 128

ACTOR_ACTIVATION_A1 = "elu"
ACTOR_ACTIVATION_A2 = "elu"
ACTOR_ACTIVATION_A3 = "tanh"

CRITIC_ACTIVATION_A1 = "elu"
CRITIC_ACTIVATION_A2 = "elu"
CRITIC_ACTIVATION_A3 = "linear"

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

NUM_TRAINING_EPISODES = 5000

BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 250000
REWARD_SCALE = 5e-3
REWARD_DISCOUNT_FACTOR = 0.001
SOFT_TARGET_UPDATE = 0.001

PARAMETER_INIT = [-3e-3, 3e-3]

def compute_weight_init(fan_in):
    return [-1/np.sqrt(fan_in), 1/np.sqrt(fan_in)]

#
env = Env(TTC_threshold)

NUM_STATES = env.num_states
print("Size of State Space ->  {}".format(NUM_STATES))
NUM_ACTIONS = env.num_actions
print("Size of Action Space ->  {}".format(NUM_ACTIONS))

upper_bound = env.action_upper_bound
lower_bound = env.action_lower_bound

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

episodes = aggregate_episodes.aggregate_episodes()
episodes = [x[0:100] for x in episodes]
# lengths = [len(x) for x in episodes]
#
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=0.1, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

#
class Buffer:
    def __init__(self, buffer_capacity=REPLAY_BUFFER_SIZE, batch_size=BATCH_SIZE):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = pd.DataFrame(columns = ["Bumper to Bumper Distance", "Following Vehicle Speed", "Relative Speed"], data = np.zeros((self.buffer_capacity, 3)))
        self.next_state_buffer = pd.DataFrame(columns = ["Bumper to Bumper Distance", "Following Vehicle Speed", "Relative Speed"], data = np.zeros((self.buffer_capacity, 3)))
        self.action_buffer = pd.Series(name = "Action", data = np.zeros(self.buffer_capacity))
        self.reward_buffer = pd.Series(name = "Reward", data = np.zeros(self.buffer_capacity))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, prev_state, action, reward, state):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer.iloc[index] = prev_state
        self.action_buffer.iloc[index] = action
        self.reward_buffer.iloc[index] = reward
        self.next_state_buffer.iloc[index] = state

        self.buffer_counter += 1
        
    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer.iloc[batch_indices])
        # print(state_batch)
        action_batch = tf.convert_to_tensor(self.action_buffer.iloc[batch_indices])
        # print(action_batch)
        reward_batch = tf.convert_to_tensor(self.reward_buffer.iloc[batch_indices])
        # print(reward_batch)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        # print(reward_batch)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer.iloc[batch_indices])
        # print(next_state_batch)

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `lr`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))



def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    
    

    inputs = layers.Input(shape=(NUM_STATES,))
    
    init_1 = tf.random_uniform_initializer(minval=-1/np.sqrt(NUM_STATES), maxval=1/np.sqrt(NUM_STATES))
    out = layers.Dense(ACTOR_SIZE_A1, activation=ACTOR_ACTIVATION_A1, kernel_initializer=init_1)(inputs)
    
    init_2 = tf.random_uniform_initializer(minval=-1/np.sqrt(ACTOR_SIZE_A1), maxval=1/np.sqrt(ACTOR_SIZE_A1))
    out = layers.Dense(ACTOR_SIZE_A2, activation=ACTOR_ACTIVATION_A2, kernel_initializer=init_2)(out)
    
    init_3 = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    outputs = layers.Dense(1, activation=ACTOR_ACTIVATION_A3, kernel_initializer=init_3)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic():
    # State as input
    state_input = layers.Input(shape=(NUM_STATES))
    
    init_1 = tf.random_uniform_initializer(minval=-1/np.sqrt(NUM_STATES), maxval=1/np.sqrt(NUM_STATES))
    state_out = layers.Dense(16, activation="elu", kernel_initializer=init_1)(state_input)
    
    init_2 = tf.random_uniform_initializer(minval=-1/np.sqrt(16), maxval=1/np.sqrt(16))
    state_out = layers.Dense(32, activation="elu", kernel_initializer=init_2)(state_out)

    # Action as input
    action_input = layers.Input(shape=(NUM_ACTIONS))
    init_3 = tf.random_uniform_initializer(minval=-1/np.sqrt(NUM_ACTIONS), maxval=1/np.sqrt(NUM_ACTIONS))
    action_out = layers.Dense(32, activation="elu", kernel_initializer=init_3)(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])
    init_4 = tf.random_uniform_initializer(minval=-1/np.sqrt(64), maxval=1/np.sqrt(64))
    out = layers.Dense(256, activation="elu", kernel_initializer=init_4)(concat)
    
    
    init_5 = tf.random_uniform_initializer(minval=-1/np.sqrt(256), maxval=1/np.sqrt(256))
    out = layers.Dense(256, activation="elu", kernel_initializer=init_5)(out)
        
    init_6 = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    outputs = layers.Dense(1, kernel_initializer=init_6)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model
    


def policy(state, noise_object):
    sampled_actions = actor_model(state)
    # print(sampled_actions)
    # print(sampled_actions)
    noise = noise_object()
    
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise
    # print(sampled_actions)

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    # print(sampled_actions)
    # print("\n")

    return [np.squeeze(legal_action)]

def get_human_reward(episode):
    previous_a = 0
    previous_follower_speed = episode ["Following Vehicle Speed"].iloc[0]
    DT = 0.1
    penalty = 100
    reward = 0
    for timestep, row in episode[1:].iterrows():
        follower_speed = row["Following Vehicle Speed"]
        a = follower_speed - previous_follower_speed
        
        b2b_distance = row["Bumper to Bumper Distance"]
        relative_speed = row["Relative Speed"]
    
        if follower_speed <= 0:
            follower_speed = 0.00001
            is_stall = 1
        else:
            is_stall = 0
        #judge collision and back
        if b2b_distance < 0:
            is_collision = 1
        else:
            is_collision = 0
        # caculate the reward
        jerk = (a - previous_a) / DT
        follower_headway = b2b_distance / follower_speed
        TTC = -b2b_distance / relative_speed  # negative sign because of relative speed sign
        
        f_jerk = -(jerk ** 2)/3600   # the maximum range is change from -3 to 3 in 0.1 s, then the jerk = 60
    
        f_acc = - a**2/60
    
        last_action = action
    
        if TTC >= 0 and TTC <= TTC_threshold:
            f_ttc = np.log(TTC/TTC_threshold) 
        else:
            f_ttc = 0
    
        mu = 0.422618  
        sigma = 0.43659
        if follower_headway <= 0:
            f_headway = -1
        else:
            f_headway = (np.exp(-(np.log(follower_headway) - mu) ** 2 / 
                                (2 * sigma ** 2)) / 
                         (follower_headway * sigma * np.sqrt(2 * np.pi)))
    
    
        # calculate the reward
        reward = f_jerk + f_ttc + f_headway - penalty * is_collision
        
        previous_follower_speed = follower_speed
        previous_a = a
    
    return reward

std_dev = 0.05
mean = 0
ou_noise = OUActionNoise(mean=float(mean) * np.ones(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()
 
target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())



critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR)
actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR)

total_episodes = 1000
# Discount factor for future rewards

gamma = 1

# Used to update target networks
tau = 0.001

buffer = Buffer(250000, 64)
#
# To store reward history of each episode
ep_reward_list = []
ep_human_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
avg_human_reward_list = []
# Takes about 4 min to train
simulated_episodes = []


#%%


for repeat in range(10):
    for i, episode in enumerate(episodes):
        # episode = episodes[0]
        prev_state = env.reset(episode)
        episodic_reward = 0
        episodic_rewards = []
        while True:
            ##################################################
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()
            # print("0: ", prev_state)
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(list(prev_state.values())), 0)
            # print(tf_prev_state)
            action = policy(tf_prev_state, ou_noise)
            # print("2", action)
            # Recieve state and reward from environment.
            state, reward, done = env.step(np.array(action)[0])
            
            # print(state, reward, done, info)
            buffer.record(prev_state, np.array(action)[0], reward, state)
            episodic_reward += reward
            episodic_rewards.append(episodic_reward)
            # print(episodic_reward)
            
            buffer.learn()
            # print(actor_model(tf_prev_state))
            update_target(target_actor.variables, actor_model.variables, tau)
            # print(actor_model(tf_prev_state))
            update_target(target_critic.variables, critic_model.variables, tau)
            # print(actor_model(tf_prev_state))
            
            # End this episode when `done` is True
            if done:
                break
            
            prev_state = state
    ######################################################
        ep_reward_list.append(episodic_reward)
        # ep_human_reward_list.append(get_human_reward(episode))
        
        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        # avg_human_reward = np.mean(ep_human_reward_list[-40:])
        print(" Episode * {} * Reward:  {} * Human Reward:  {}".format(i, episodic_reward, get_human_reward(episode)))
        print(get_human_reward(env.state_history))
        avg_reward_list.append(avg_reward)
        # avg_human_reward_list.append(avg_human_reward)
        simulated_episodes.append(env.state_history)
        
        if (i % 10 == 0):
            animate.animate(episode, env.state_history, episodic_rewards, i)
            plt.plot(ep_reward_list, label = "Machine")
            # plt.plot(avg_human_reward_list, label = "Human")
            plt.legend()
            plt.xlabel("Episode")
            plt.ylabel("Epsiodic Reward")
            plt.show()
            plt.close()

    pygame.quit()
#%%
# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list, label = "Machine")
# plt.plot(avg_human_reward_list, label = "Human")
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
plt.close()



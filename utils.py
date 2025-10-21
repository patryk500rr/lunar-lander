import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from IPython.display import clear_output
from collections import deque, namedtuple
import random
import time
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import utils


# Helper functions
def get_new_eps(eps, eps_min, eps_decay_rate):
    eps = max(eps_min, eps * eps_decay_rate)
    return eps


def soft_update_target(tau, target_q_network, q_network):
    """
        Updates target_q_network weights using soft_update rule

        Args:
            -tau (float): hyperparameter- how aggresive we want to be with updates
            -target_q_network (tf model)
            -q_network (tf_model)
        Returns:
            -None
    """
    for target_weights, q_weights in zip(target_q_network.weights, q_network.weights):
        target_weights.assign(tau * q_weights + (1 - tau) * target_weights)



def get_action(q_values, eps, n_actions=4):
    """
        Chooses an action given by q_values using e-greedy policy

        Args:
            q_values (tf tensor): propabilities for each action at given state
            eps (float): how much we want to explore or exploate
        return:
            action (int)
    """
    if random.random() > eps:
        action = np.argmax(q_values.numpy()[0])
        return action
    else:
        return random.choice(np.arange(n_actions))


def get_minibatch(memory_buffer, minibatch_size):
    """
        Return a random sample from memory_buffer

        Args:
            -memory_buffer (deque): states, actions, rewards, next_states, done_vals
            -minibatch_size (int): size of random sample that will be returned
        Returns:
            -Experiences (tuple, dtype: tf.float32): states, actions, rewards, next_states, done_vals    | of len minibatch_size
    """
    experiences = random.sample(memory_buffer, k=minibatch_size)
    
    states = tf.convert_to_tensor(
        np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
    )   
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
    )  
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
    )  
    next_states = tf.convert_to_tensor(
        np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
    )  
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences if e is not None]).astype(np.uint8), dtype=tf.float32
    )  

    return (states, actions, rewards, next_states, done_vals)



def check_update_conditions(t, memory_buffer, num_steps_for_update, minibatch_size):
    """
        Checks if we can update weights at given iteration

        Args:
            -t: num step at given episode
            -memory_buffer (deque): states, actions, rewards, next_states, done_vals
            -num_steps_for_update (int): hyperparameter
            -minibatch_size (int)
        Returns:
            -(bool): whether we can train agent at given t
    """
    m = len(memory_buffer)
    if (t + 1) % num_steps_for_update == 0 and m > minibatch_size:
        return True
    else:
        return False
    
    
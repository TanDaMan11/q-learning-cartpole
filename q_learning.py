# First, install the necessary libraries
!pip install gym

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Set up parameters
n_actions = env.action_space.n
n_states = 24  # For simplicity, we will discretize the state space into 24 bins
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration-exploitation tradeoff
episodes = 1000  # Number of training episodes

# Discretize the state space (CartPole's continuous state space)
def discretize_state(state):
    state_bins = [
        np.linspace(-2.4, 2.4, n_states),
        np.linspace(-3.0, 3.0, n_states),
        np.linspace(-0.5, 0.5, n_states),
        np.linspace(-3.0, 3.0, n_states)
    ]
    state_discretized = [
        np.digitize(state[0], state_bins[0]),
        np.digitize(state[1], state_bins[1]),
        np.digitize(state[2], state_bins[2]),
        np.digitize(state[3], state_bins[3])
    ]
    return tuple(state_discretized)

# Initialize Q-table
Q = np.zeros([n_states] * 4 + [n_actions])  # Q[state][action]

# Define epsilon-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q[state])  # Exploit

# Train the agent
episode_rewards = []

for episode in range(episodes):
    state = discretize_state(env.reset())
    total_reward = 0
    done = False
    
    while not done:
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        
        # Q-learning update rule
        Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        
        state = next_state
        total_reward += reward
        
    episode_rewards.append(total_reward)
    
    if episode % 100 == 0:
        print(f"Episode {episode}/{episodes}, Total Reward: {total_reward}")

# Plot the rewards over time
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-learning Training Progress')
plt.show()

# Test the trained agent
print("Testing trained agent:")
state = discretize_state(env.reset())
done = False
total_reward = 0

while not done:
    action = np.argmax(Q[state])  # Exploit the learned Q-values
    next_state, reward, done, _, _ = env.step(action)
    next_state = discretize_state(next_state)
    state = next_state
    total_reward += reward

print(f"Test Total Reward: {total_reward}")

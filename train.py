import gymnasium as gym
import numpy as np
import pygame
import random
from traffic_env import TrafficEnvironment

# Hyperparameters
alpha = 0.1    # Learning rate
gamma = 0.99   # Discount factor
epsilon = 1.0  # Epsilon for exploration (initially 100% exploration)
epsilon_min = 0.01  # Minimum epsilon
epsilon_decay = 0.99995  # Decay factor for epsilon


# Function to convert Box observation/action space to a single discrete index
def flatten_space(space, values):
    low = space.low
    shape = space.shape
    flat_value = 0
    for i, (val, lo) in enumerate(zip(values, low)):
        flat_value *= (space.high[i] - lo + 1)
        flat_value += (val - lo)
    return int(flat_value)

# Initialize the Q-table with flattened observation and action space sizes
def init_q_table(observation_space, action_space):
    obs_size = np.prod(observation_space.high - observation_space.low + 1)
    act_size = np.prod(action_space.high - action_space.low + 1)
    return np.zeros((obs_size, act_size))

# Choose action using epsilon-greedy policy
def choose_action(state, q_table, action_space, epsilon):
    if np.random.rand() < epsilon:  # Exploration
        # Randomly choose each action component between the low and high bounds of action_space
        action_values = [random.randint(int(low), int(high)) for low, high in zip(action_space.low, action_space.high)]
        return flatten_space(action_space, action_values)
    else:  # Exploitation
        return np.argmax(q_table[state])

# Update Q-values
def update_q_table(q_table, state, action, reward, next_state, done):
    best_next_action = np.argmax(q_table[next_state])
    q_target = reward + gamma * q_table[next_state][best_next_action] * (1 - int(done))
    q_table[state, action] = q_table[state, action] + alpha * (q_target - q_table[state, action])

def main():
    env = TrafficEnvironment(paramsfile="params_files/params_example.json")
    env.do_render = True

    # Discretize the observation and action spaces
    observation_space = env.observation_space
    action_space = env.action_space

    # Initialize Q-table
    # q_table = init_q_table(observation_space, action_space)
   
    q_table = np.load('qtable_saved3.npy', allow_pickle=True)
    
    observation, info = env.reset()
    state = flatten_space(observation_space, observation)  # Flatten observation into discrete state

    global epsilon  # Make epsilon modifiable inside the loop
    epsilon = 0.01
    episodic_reward = 0
    count_completed = 0
    ep_rewards = []
    for episode in range(10000000):
        if env.do_render:        
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

        action = choose_action(state, q_table, action_space, epsilon)
        
        # Unflatten action index to get corresponding action values
        action_unflattened = np.unravel_index(action, [int(high - low + 1) for low, high in zip(action_space.low, action_space.high)])
        action_unflattened = [int(low + val) for val, low in zip(action_unflattened, action_space.low)]  # Convert back to original scale
        
        next_observation, reward_tot, done, info = env.step(action_unflattened)
        reward = reward_tot[0]+reward_tot[1]
        next_state = flatten_space(observation_space, next_observation)  # Flatten next observation

        update_q_table(q_table, state, action, reward, next_state, done)

        # print(f"Action: {action_unflattened}, State: {state}, Reward: {reward}, New State: {next_state}")
        episodic_reward += reward
        # print(episodic_reward)
        
        state = next_state

        if env.do_render:
            env.render()

        if done:
            observation, info = env.reset()
            state = flatten_space(observation_space, observation)  # Reset state
            epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay epsilon
            count_completed +=1
            ep_rewards.append(episodic_reward)
            if len(ep_rewards) > 200:
                last_window_reward = np.mean(ep_rewards[-200:])
            else:
                last_window_reward = np.mean(ep_rewards)
            episodic_reward = 0
            if count_completed%100 == 0:
                print(f"Ep: {episode}, completed: {count_completed}, Ep.reward: {last_window_reward}, eps: {epsilon}")
        
            
            # with open('qtable.npy', 'wb') as f:
            #     np.save(f, q_table)
    

    env.close()

if __name__ == "__main__":
    main()

import numpy as np
import gymnasium as gym
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
#import torch
#run pip install gym and pip install gym[box2d] in terminal
seed_num = 42


#Define Enviornments with wind of varying levels
env_no_wind = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)
env_wind15 = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=True, wind_power=15.0, turbulence_power=1.5)
env_wind5 = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=True, wind_power=5.0, turbulence_power=0.5)
env_wind10 = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=True, wind_power=10.0, turbulence_power=1.0)
env_wind20 = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=True, wind_power=20.0, turbulence_power=2.0)

env_wind5.action_space.seed(seed_num) 
env_wind15.action_space.seed(seed_num) 
env_wind20.action_space.seed(seed_num) 
env_wind10.action_space.seed(seed_num) 
env_no_wind.action_space.seed(seed_num) 

envs = [env_no_wind, env_wind5, env_wind10, env_wind15, env_wind20]

#######

# Neural Network for Q-function approximation
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# Replay buffer to store experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Deep Q-Learning function
def train_dqn(env, episodes, gamma=0.99, lr=1e-3, batch_size=64, buffer_capacity=100000, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    dqn = DQN(input_dim, output_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    criterion = nn.MSELoss()
    buffer = ReplayBuffer(buffer_capacity)
    epsilon = epsilon_start

    rewards = []  # Initialize as a Python list
    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
    
        for t in range(1000):  # max steps per episode
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(dqn(state)).item()
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            buffer.add(state, action, reward, next_state, done)
    
            state = next_state
            total_reward += reward
    
            # Train the DQN
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards_batch, next_states, dones = zip(*batch)
    
                states = torch.cat(states).float()
                actions = torch.tensor(actions).unsqueeze(1)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).unsqueeze(1)
                next_states = torch.cat(next_states).float()
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
    
                q_values = dqn(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q_values = dqn(next_states).max(1, keepdim=True)[0]
                    target_q_values = rewards_batch + gamma * max_next_q_values * (1 - dones)
    
                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
    
        rewards.append(total_reward)  # Append total_reward to Python list
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode+1}, Reward: {total_reward}, Epsilon: {epsilon}")
    
    return rewards, dqn


# handle dd
def train_ddqn(env, episodes, gamma=0.99, lr=1e-3, batch_size=64, buffer_capacity=100000, 
               epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995, target_update_freq=10):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    # Initialize both the Q-network and the Target network
    q_network = DQN(input_dim, output_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    target_network = DQN(input_dim, output_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    target_network.load_state_dict(q_network.state_dict())  # Initialize target with Q-network weights
    target_network.eval()  # Target network in evaluation mode
    
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    criterion = nn.MSELoss()
    buffer = ReplayBuffer(buffer_capacity)
    epsilon = epsilon_start

    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0

        for t in range(1000):  # max steps per episode
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(q_network(state)).item()
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            buffer.add(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Train the Q-network
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards_batch, next_states, dones = zip(*batch)

                states = torch.cat(states).float()
                actions = torch.tensor(actions).unsqueeze(1)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).unsqueeze(1)
                next_states = torch.cat(next_states).float()
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                # Q-values for chosen actions
                q_values = q_network(states).gather(1, actions)

                
                with torch.no_grad():
                    # ddqn adjustments
                    next_actions = torch.argmax(q_network(next_states), dim=1, keepdim=True)
                    next_q_values = target_network(next_states).gather(1, next_actions)
                    target_q_values = rewards_batch + gamma * next_q_values * (1 - dones)

                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        rewards.append(total_reward)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode+1}, Reward: {total_reward}, Epsilon: {epsilon}")

        # Update target network periodically
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

    return rewards, q_network

######################

# Dueling dqn architechure
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single output for V(s)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)  # Outputs for A(s, a)
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
# literally vanilla but with dueling architecture
def train_dueling_dqn(env, episodes, gamma=0.99, lr=1e-3, batch_size=64, buffer_capacity=100000, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    dqn = DuelingDQN(input_dim, output_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    criterion = nn.MSELoss()
    buffer = ReplayBuffer(buffer_capacity)
    epsilon = epsilon_start

    rewards = []  # Initialize as a Python list
    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
    
        for t in range(1000):  # max steps per episode
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(dqn(state)).item()
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            buffer.add(state, action, reward, next_state, done)
    
            state = next_state
            total_reward += reward
    
            # Train the DQN
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards_batch, next_states, dones = zip(*batch)
    
                states = torch.cat(states).float()
                actions = torch.tensor(actions).unsqueeze(1)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).unsqueeze(1)
                next_states = torch.cat(next_states).float()
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
    
                q_values = dqn(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q_values = dqn(next_states).max(1, keepdim=True)[0]
                    target_q_values = rewards_batch + gamma * max_next_q_values * (1 - dones)
    
                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
    
        rewards.append(total_reward)  # Append total_reward to Python list
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode+1}, Reward: {total_reward}, Epsilon: {epsilon}")
    
    return rewards, dqn




if __name__ == "__main__":
    # Define paths for results and weights
    results_folder = "experiments"
    weights_folder = "weights"
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(weights_folder, exist_ok=True)

    # Map algorithm names to training functions
    algorithms = {
        "DQN": train_dqn,
        "DDQN": train_ddqn,
        "DuelingDQN": train_dueling_dqn
    }

    # Run experiments
    for env_idx, env in enumerate(envs):
        env_name = f"env_wind_{env_idx}"
        print(f"Running experiments for {env_name}...")

        for algo_name, algo_function in algorithms.items():
            print(f"Training {algo_name} on {env_name}...")
            rewards, model = algo_function(env, episodes=1000)  # Adjust episodes as needed

            # Save rewards
            rewards_path = os.path.join(results_folder, f"{env_name}_{algo_name}_rewards.txt")
            with open(rewards_path, "w") as f:
                for reward in rewards:
                    f.write(f"{reward}\n")

            # Save model weights
            weights_path = os.path.join(weights_folder, f"{env_name}_{algo_name}_weights.pth")
            torch.save(model.state_dict(), weights_path)

            print(f"{algo_name} on {env_name} completed. Rewards and weights saved.")

    print("All experiments completed.")
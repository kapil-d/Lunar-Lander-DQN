import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from ll_algorithms import envs, train_dqn, train_ddqn, train_dueling_dqn, DQN, DuelingDQN
import torch
import random

seed_num = 42


#Define Enviornments with wind of varying levels
env_no_wind = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode='human')
env_wind15 = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=True, wind_power=15.0, turbulence_power=1.5, render_mode='human')
env_wind5 = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=True, wind_power=5.0, turbulence_power=0.5, render_mode='human')
env_wind10 = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=True, wind_power=10.0, turbulence_power=1.0, render_mode='human')
env_wind20 = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=True, wind_power=20.0, turbulence_power=2.0, render_mode='human')

env_wind5.action_space.seed(seed_num) 
env_wind15.action_space.seed(seed_num) 
env_wind20.action_space.seed(seed_num) 
env_wind10.action_space.seed(seed_num) 
env_no_wind.action_space.seed(seed_num) 

envs = [env_no_wind, env_wind5, env_wind10, env_wind15, env_wind20]
def load_model(model_class, input_dim, output_dim, weights_path):
    """
    Reinitializes a model and loads weights from a .pth file.
    
    Parameters:
        model_class: The class of the model (e.g., DQN, DuelingDQN).
        input_dim: The input dimension for the model.
        output_dim: The output dimension for the model.
        weights_path: Path to the saved weights (.pth file).
        
    Returns:
        A model instance with loaded weights.
    """
    # Initialize the model
    model = model_class(input_dim, output_dim)
    
    # Load the saved weights
    model.load_state_dict(torch.load(weights_path))
    
    # Set the model to evaluation mode
    #model.eval()
    
    return model

# Example: Reloading a DQN model
input_dim = 8  # Adjust based on your environment's observation space
output_dim = 4  # Adjust based on your environment's action space
weights_path = "weights/env_wind_3_DQN_weights.pth"  # Path to the saved weights

# Reinitialize the model
model = load_model(DQN, input_dim, output_dim, weights_path)

# visualize performances
def select_action(state, model, epsilon=0.0):
    if random.random() < epsilon:  # Explore
        return env.action_space.sample()
    else:  # Exploit
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()
        return action

env = envs[3]
rewards = 0
state, _ = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
for _ in range(10000):
    action = select_action(state, model, epsilon=0.0)  # Greedy action
    next_state, reward, terminated, truncated, _ = env.step(action)
    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
    state = next_state
    rewards += reward

    if terminated or truncated:
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        print(f'rewards = {rewards}')
        rewards = 0
    

env.close()
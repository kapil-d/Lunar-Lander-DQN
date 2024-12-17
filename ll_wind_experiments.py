import os
import torch
from ll_algorithms import envs, train_dqn, train_ddqn, train_dueling_dqn

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
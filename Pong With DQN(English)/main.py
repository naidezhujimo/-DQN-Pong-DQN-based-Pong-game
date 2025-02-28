import gymnasium as gym
from collections import namedtuple
import time
import torch
import ale_py
from tensorboardX import SummaryWriter
import atari_wrappers
from agent import DQNAgent

DQN_HYPERPARAMS = {
    'dueling': False,
    'noisy_net': False,
    'double_DQN': False,
    'n_multi_step': 2,
    'buffer_start_size': 10001,
    'buffer_capacity': 15000,
    'epsilon_start': 1.0,
    'epsilon_decay': 10**5,
    'epsilon_final': 0.02,
    'learning_rate': 5e-5,
    'gamma': 0.99,
    'n_iter_update_target': 1000
}

BATCH_SIZE = 64  # The size of each sampled batch
MAX_N_GAMES = 3000  # The maximum number of games to run

ENV_NAME = "ALE/Pong-v5"  # The environment name; here, we use the Atari game Pong
SAVE_VIDEO = True  # Whether to save game videos
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_WRITER = True  # Whether to use TensorBoard to record the training process

LOG_DIR = 'content/runs'
name = '_'.join([str(k)+'.'+str(v) for k,v in DQN_HYPERPARAMS.items()])
name = 'prv'

if __name__ == '__main__':
    # Create the environment
    env = atari_wrappers.make_env(ENV_NAME, render_mode="rgb_array")  # Pass the environment object

    if SAVE_VIDEO:
        # Use the RecordVideo wrapper to record videos
        env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda x: x % 10 == 0)

    obs, _ = env.reset()  # Get the initial observation and info

    # TensorBoard is used to record statistical information during the training process.
    writer = SummaryWriter(log_dir=LOG_DIR+'/'+name + str(time.time())) if SUMMARY_WRITER else None

    print('Hyperparams:', DQN_HYPERPARAMS)

    # Create the agent
    agent = DQNAgent(env, device=DEVICE, summary_writer=writer, hyperparameters=DQN_HYPERPARAMS)

    # Initialize the number of games and iterations
    n_games = 0
    n_iter = 0

    # Run the game for MAX_N_GAMES times
    while n_games < MAX_N_GAMES:
        action = agent.act_eps_greedy(obs)  # Select an action using ε-greedy strategy
        # Execute the action and get the new observation, reward, termination flag, and truncation flag
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Add the experience (observation, action, new observation, reward, termination flag) to the replay buffer
        agent.add_env_feedback(obs, action, new_obs, reward, done)
        # Sample from the replay buffer and optimize the network
        agent.sample_and_optimize(BATCH_SIZE)

        obs = new_obs  # Update the current observation
        if done:
            n_games += 1  # Increment the number of games
            agent.print_info()  # Print training information
            agent.reset_stats()  # Reset the agent's statistics
            obs, _ = env.reset()  # Reset the environment and get the new observation and info

    writer.close()  # Close the SummaryWriter
    env.close()  # Ensure the environment is closed to save the recorded videos
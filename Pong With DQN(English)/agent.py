import numpy as np
from collections import namedtuple
import time
from central_control import CentralControl
from buffers import ReplayBuffer


class DQNAgent():
    rewards = []  # Store the total reward for each game
    total_reward = 0  # Total reward for the current game
    birth_time = 0  # Time when the agent was created
    n_iter = 0  # Total number of iterations
    n_games = 0  # Total number of games played
    ts_frame = 0  # Timestamp for calculating FPS
    ts = time.time()  # Initial timestamp

    # Named tuple for storing experiences (observation, action, new observation, reward, done flag)
    Memory = namedtuple('Memory', ['obs', 'action', 'new_obs', 'reward', 'done'], rename=False)

    def __init__(self, env, device, hyperparameters, summary_writer=None):
        """
        env: Environment object, providing observation space and action space
        device: Device to run on (e.g., CPU or GPU)
        hyperparameters: Dictionary of hyperparameters, including discount factor, learning rate, etc.
        summary_writer: Optional object for recording training statistics
        """
        # Initialize CentralControl, responsible for managing network training and updates
        self.cc = CentralControl(env.observation_space.shape, env.action_space.n, hyperparameters['gamma'], hyperparameters['n_multi_step'], hyperparameters['double_DQN'],
                                 hyperparameters['noisy_net'], hyperparameters['dueling'], device)

        self.cc.set_optimizer(hyperparameters['learning_rate'])

        self.birth_time = time.time()  # Record the time when the agent was created

        self.iter_update_target = hyperparameters['n_iter_update_target']  # Interval for updating the target network
        self.buffer_start_size = hyperparameters['buffer_start_size']  # Minimum buffer size required before starting training

        self.epsilon_start = hyperparameters['epsilon_start']  # Initial exploration rate
        self.epsilon = hyperparameters['epsilon_start']  # Current exploration rate
        self.epsilon_decay = hyperparameters['epsilon_decay']  # Decay rate for the exploration rate
        self.epsilon_final = hyperparameters['epsilon_final']  # Final exploration rate

        self.accumulated_loss = []  # Store losses during training
        self.device = device

        # Replay buffer for storing and sampling experiences
        self.replay_buffer = ReplayBuffer(hyperparameters['buffer_capacity'], hyperparameters['n_multi_step'], hyperparameters['gamma'])
        # Optional object for recording training statistics
        self.summary_writer = summary_writer

        # Whether to use a noisy network
        self.noisy_net = hyperparameters['noisy_net']

        # Environment object
        self.env = env

    # Select the optimal action
    def act(self, obs):
        return self.cc.get_max_action(obs)

    # ε-greedy strategy
    def act_eps_greedy(self, obs):
        if self.noisy_net:
            return self.act(obs)
        # With probability epsilon, select a random action; otherwise, select the optimal action
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.act(obs)

    # Add feedback from the environment
    def add_env_feedback(self, obs, action, new_obs, reward, done):
        # Ensure obs and new_obs are NumPy arrays
        obs_arr = np.array(obs, dtype=np.float32)
        new_obs_arr = np.array(new_obs, dtype=np.float32)

        # Check for NaN or Inf values
        if not np.isfinite(obs_arr).all() or not np.isfinite(new_obs_arr).all():
            print(f"NaN or Inf found in observation: {obs} or {new_obs}")

        # Clip the reward to the range [-1, 1]
        reward = np.clip(reward, -1, 1)

        # Create a memory and update the buffer
        new_memory = self.Memory(obs=obs_arr, action=action, new_obs=new_obs_arr, reward=reward, done=done)
        self.replay_buffer.append(new_memory)

        # Update iteration count, exploration rate, and total reward for the current game
        self.n_iter += 1
        self.epsilon = max(self.epsilon_final, self.epsilon_start - self.n_iter / self.epsilon_decay)
        self.total_reward += reward

    # Sample and optimize
    def sample_and_optimize(self, batch_size):
        # If the replay buffer size is greater than buffer_start_size, sample batch_size experiences and optimize using CentralControl's optimize method
        if len(self.replay_buffer) > self.buffer_start_size:
            mini_batch = self.replay_buffer.sample(batch_size)
            l_loss = self.cc.optimize(mini_batch)
            # Append the optimized loss to accumulated_loss
            self.accumulated_loss.append(l_loss)

        # Update the target network if the current iteration is a multiple of iter_update_target
        if self.n_iter % self.iter_update_target == 0:
            self.cc.update_target()

    # Reset statistics
    def reset_stats(self):
        self.rewards.append(self.total_reward)  # Append the total reward for the current game to the rewards list
        self.total_reward = 0  # Reset the total reward for the current game
        self.accumulated_loss = []  # Reset the accumulated loss list
        self.n_games += 1  # Increment the game count

    # Print information
    def print_info(self):
        # Calculate FPS (frames per second)
        fps = (self.n_iter - self.ts_frame) / (time.time() - self.ts)

        # Calculate the mean loss
        mean_loss = np.mean(self.accumulated_loss) if self.accumulated_loss else 0.0

        mean_rew_40 = np.mean(self.rewards[-40:]) if len(self.rewards) > 40 else np.mean(self.rewards)
        mean_rew_10 = np.mean(self.rewards[-10:]) if len(self.rewards) > 10 else np.mean(self.rewards)

        print(f"{self.n_iter} {self.n_games} rew:{self.total_reward} mean_rew_40:{mean_rew_40:.2f} mean_rew_10:{mean_rew_10:.2f} eps:{self.epsilon:.2f} fps:{fps:.0f} loss:{mean_loss:.4f}")

        self.ts_frame = self.n_iter
        self.ts = time.time()
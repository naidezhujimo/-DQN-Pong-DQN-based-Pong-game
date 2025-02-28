import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from neural_net import DQN, DuelingDQN

# The control center of the entire reinforcement learning system, responsible for managing network initialization, optimization, target network updates, and action selection
class CentralControl():
    def __init__(self, observation_space_shape, action_space_shape, gamma, n_multi_step, double_DQN, noisy_net, dueling, device):
        """
        observation_space_shape: The shape of the observation space
        action_space_shape: The size of the action space
        gamma: Discount factor
        n_multi_step: Number of steps for multi-step returns
        double_DQN: Whether to use Double DQN
        noisy_net: Whether to use a noisy network
        dueling: Whether to use Dueling DQN
        device: The device to run on
        """
        if dueling:
            # The target network provides stable Q-value estimates, while the moving network is used for training
            self.target_nn = DuelingDQN(observation_space_shape, action_space_shape).to(device)
            self.moving_nn = DuelingDQN(observation_space_shape, action_space_shape).to(device)
        else:
            self.target_nn = DQN(observation_space_shape, action_space_shape, noisy_net).to(device)
            self.moving_nn = DQN(observation_space_shape, action_space_shape, noisy_net).to(device)

        self.device = device
        self.gamma = gamma
        self.n_multi_step = n_multi_step
        self.double_DQN = double_DQN

    # Set the optimizer
    def set_optimizer(self, learning_rate):
        self.optimizer = optim.Adam(self.moving_nn.parameters(), lr=learning_rate)

    # Optimization method
    def optimize(self, mini_batch):
        self.optimizer.zero_grad()  # Zero the gradients
        loss = self._calulate_loss(mini_batch)  # Calculate the loss function
        loss.backward()  # Backpropagation
        torch.nn.utils.clip_grad_norm_(self.moving_nn.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()  # Update network parameters
        return loss.item()

    def update_target(self):
        # Copy the weights of the moving network to the target network
        self.target_nn.load_state_dict(self.moving_nn.state_dict())
        self.target_nn = self.moving_nn

    # Select the optimal action
    def get_max_action(self, obs):
        # Convert the observation obs to a tensor
        state_t = torch.tensor(np.array([obs])).to(self.device)
        # Calculate Q-values using the moving network
        q_values_t = self.moving_nn(state_t)
        # Select the action with the highest Q-value
        _, act_t = torch.max(q_values_t, dim=1)
        return int(act_t.item())

    # Calculate the loss function
    def _calulate_loss(self, mini_batch):
        # Extract states, actions, next states, rewards, and done flags from mini_batch
        states, actions, next_states, rewards, dones = mini_batch

        # Convert data to tensors and move to the specified device
        states_t = torch.as_tensor(states, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(dones, dtype=torch.uint8, device=self.device)

        # Calculate Q-values for the current states
        """
        self.moving_nn(states_t): Calculate Q-values for all actions
        .gather(1, actions_t[:,None]): Select Q-values for the actions taken
        .squeeze(-1): Remove the extra dimension
        """
        state_action_values = self.moving_nn(states_t).gather(1, actions_t[:, None]).squeeze(-1)

        """
        Calculate Q-values for the next states:
            If using Double DQN:
                Use the moving network to select the optimal action
                Use the target network to calculate the Q-value for that action
            If not using Double DQN:
                Directly use the target network to calculate the maximum Q-value for the next state
        """
        if self.double_DQN:
            double_max_action = self.moving_nn(next_states_t).max(1)[1]
            double_max_action = double_max_action.detach()
            target_output = self.target_nn(next_states_t)
            next_state_values = torch.gather(target_output, 1, double_max_action[:, None]).squeeze(-1)
        else:
            next_state_values = self.target_nn(next_states_t).max(1)[0]

        # Detach the next state Q-values from the computation graph to prevent backpropagation
        next_state_values = next_state_values.detach()
        # Calculate the target Q-values using the Bellman equation
        expected_state_action_values = rewards_t + (self.gamma ** self.n_multi_step) * next_state_values
        # Calculate the mean squared error loss
        return nn.MSELoss()(state_action_values, expected_state_action_values)
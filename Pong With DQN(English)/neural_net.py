import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter, init
from torch.nn import functional as F
import math

# Noisy Network
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        """
        - in_features: Number of input features
        - out_features: Number of output features
        - sigma_init: Initial value of the noise standard deviation
        - bias: Whether to use bias
        """
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_init = sigma_init  # Save the initial value of the noise standard deviation

        # Learnable parameter representing the noise standard deviation of the weights
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        # Register a buffer epsilon_weight to store the noise of the weights
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        if bias:  # Define noise standard deviation and noise buffer for bias
            self.sigma_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    # Initialize weights and biases
    def reset_parameters(self):
        if hasattr(self, 'sigma_bias'):
            init.constant_(self.sigma_bias, self.sigma_init)
            init.constant_(self.sigma_weight, self.sigma_init)

        # Initialize weights and biases using a uniform distribution
        std = math.sqrt(3 / self.in_features)
        init.uniform_(self.weight, -std, std)
        init.uniform_(self.bias, -std, std)

    def forward(self, input):
        if self.bias is not None:
            self.epsilon_bias.data.normal_()  # Generate normally distributed noise
            # Multiply the noise by sigma_bias and add it to the bias
            bias = self.bias + self.sigma_bias * self.epsilon_bias
        else:
            bias = self.bias  # If bias is not used, directly use the original bias

        # Generate weight noise and add it to the weights
        self.epsilon_weight.data.normal_()
        weight = self.weight + self.sigma_weight * self.epsilon_weight

        return F.linear(input, weight, bias)  # Compute the linear transformation, applying the noisy weights and bias to the input

# Separate modeling of state value function (V) and advantage function (A)
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        """
        input_shape: Shape of the input tensor
        n_actions: Size of the action space
        """
        super(DuelingDQN, self).__init__()

        # Define convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        # Calculate the output size of the convolutional layers
        conv_out_size = self._get_conv_out(input_shape)

        # Define the advantage function branch
        self.fc_a = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions))

        # Define the state value function branch
        self.fc_v = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1))

    # Method to calculate convolutional output size
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        # Calculate the output size of the convolutional layers by passing a zero tensor
        return int(np.prod(o.size()))

    def forward(self, x):
        # Perform convolution on the input x and flatten the result
        batch_size = x.size()[0]
        conv_out = self.conv(x).view(batch_size, -1)

        # Calculate the advantage function and state value function separately
        adv = self.fc_a(conv_out)
        val = self.fc_v(conv_out)

        # Output the final Q-values
        return val + adv - torch.mean(adv, dim=1, keepdim=True)

# Implementation of the Deep Q-Network architecture
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, noisy_net):
        """
        input_shape: Shape of the input tensor
        n_actions: Size of the action space
        noisy_net: Whether to use a noisy network
        """
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        conv_out_size = self._get_conv_out(input_shape)

        if noisy_net:
            self.fc = nn.Sequential(
                NoisyLinear(conv_out_size, 512),
                nn.ReLU(),
                NoisyLinear(512, n_actions))
        else:
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions))

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        conv_out = self.conv(x).view(batch_size, -1)
        return self.fc(conv_out)
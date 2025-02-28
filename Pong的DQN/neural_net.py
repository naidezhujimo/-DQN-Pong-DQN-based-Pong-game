import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter, init
from torch.nn import functional as F
import math

# 噪声网络
class NoisyLinear(nn.Linear):
	def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
		"""
		- in_features: 输入特征数
		- out_features: 输出特征数
		- sigma_init: 噪声的标准差初始化值
		- bias: 是否使用偏置
		"""
		super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
		self.sigma_init = sigma_init # 保存噪声的标准差初始化值

		# 可学习的参数,表示权重的噪声标准差
		self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
		# 注册一个缓冲区epsilon_weight,用于存储权重的噪声
		self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
		if bias: # 定义偏置的噪声标准差和噪声缓冲区
			self.sigma_bias = Parameter(torch.Tensor(out_features))
			self.register_buffer('epsilon_bias', torch.zeros(out_features))
		self.reset_parameters() 

	# 初始化权重和偏置
	def reset_parameters(self):
		if hasattr(self, 'sigma_bias'):
			init.constant_(self.sigma_bias, self.sigma_init)
			init.constant_(self.sigma_weight, self.sigma_init)

		# 使用均匀分布初始化权重和偏置
		std = math.sqrt(3/self.in_features)
		init.uniform_(self.weight, -std, std)
		init.uniform_(self.bias, -std, std)

	def forward(self, input):
		if self.bias is not None: 
			self.epsilon_bias.data.normal_() # 生成正态分布的噪声
			# 将噪声乘以sigma_bias并加到偏置上
			bias = self.bias + self.sigma_bias*self.epsilon_bias 
		else:
			bias = self.bias # 如果不使用偏置项,直接使用原始偏置

		# 生成权重噪声并添加到权重上
		self.epsilon_weight.data.normal_()
		weight = self.weight + self.sigma_weight*self.epsilon_weight

		return F.linear(input, weight, bias) # 计算线性变换,将噪声后的权重和偏置应用到输入上

# 将状态值函数(V)和优势函数(A)分开建模
class DuelingDQN(nn.Module):
	def __init__(self, input_shape, n_actions):
		"""
		input_shape: 输入张量的形状
		n_actions: 动作空间的大小
		"""
		super(DuelingDQN, self).__init__()

		# 定义卷积层
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

		# 计算卷积层的输出大小
		conv_out_size = self._get_conv_out(input_shape)

		# 定义优势函数分支
		self.fc_a = nn.Sequential(
			nn.Linear(conv_out_size, 512),
			nn.ReLU(),
			nn.Linear(512, n_actions))

		# 定义状态值函数分支
		self.fc_v = nn.Sequential(
			nn.Linear(conv_out_size, 512),
			nn.ReLU(),
			nn.Linear(512, 1))

	# 卷积输出计算方法
	def _get_conv_out(self, shape):
		o = self.conv(torch.zeros(1, *shape))
		# 通过输入一个全零张量,计算卷积层的输出大小
		return int(np.prod(o.size())) 

	def forward(self, x):
		# 对输入 x 进行卷积操作,并将结果展平
		batch_size = x.size()[0]
		conv_out = self.conv(x).view(batch_size, -1) 

		# 分别计算优势函数和状态值函数
		adv = self.fc_a(conv_out)
		val = self.fc_v(conv_out)

		# 输出最终的 Q 值
		return val + adv - torch.mean(adv, dim=1, keepdim=True)

# 实现了 Deep Q-Network 架构
class DQN(nn.Module):

	def __init__(self, input_shape, n_actions, noisy_net):
		"""
		input_shape: 输入张量的形状
		n_actions: 动作空间的大小
		noisy_net: 是否使用噪声网络
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
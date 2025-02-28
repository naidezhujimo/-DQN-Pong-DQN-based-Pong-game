import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from neural_net import DQN, DuelingDQN

# 是整个强化学习系统的控制中心,负责管理网络的初始化、优化、目标网络更新以及动作选择
class CentralControl():
	def __init__(self, observation_space_shape, action_space_shape, gamma, n_multi_step, double_DQN, noisy_net, dueling, device):
		"""
		observation_space_shape: 状态空间的形状
		action_space_shape: 动作空间的大小
		gamma: 折扣因子
		n_multi_step: 多步回报的步数
		double_DQN: 是否使用Double DQN
		noisy_net: 是否使用噪声网络
		dueling:是否使用Dueling DQN
		device:运行设备
		"""
		if dueling:
			# 目标网络用于提供稳定的Q值估计,而移动网络用于训练
			self.target_nn = DuelingDQN(observation_space_shape, action_space_shape).to(device)
			self.moving_nn = DuelingDQN(observation_space_shape, action_space_shape).to(device)
		else:
			self.target_nn = DQN(observation_space_shape, action_space_shape, noisy_net).to(device)
			self.moving_nn = DQN(observation_space_shape, action_space_shape, noisy_net).to(device)

		self.device = device
		self.gamma = gamma
		self.n_multi_step = n_multi_step
		self.double_DQN = double_DQN

	# 设置优化器
	def set_optimizer(self, learning_rate):
		self.optimizer = optim.Adam(self.moving_nn.parameters(), lr=learning_rate)

	# 优化方法
	def optimize(self, mini_batch):
		self.optimizer.zero_grad() # 清零梯度
		loss = self._calulate_loss(mini_batch) # 计算损失函数
		loss.backward() # 反向传播
		torch.nn.utils.clip_grad_norm_(self.moving_nn.parameters(), max_norm=1.0) # 梯度裁剪
		self.optimizer.step() # 更新网络参数
		return loss.item()

	def update_target(self):
		# 将移动网络的权重复制到目标网络
		self.target_nn.load_state_dict(self.moving_nn.state_dict())
		self.target_nn = self.moving_nn

	# 选择最优动作
	def get_max_action(self, obs):
		# 将状态obs转换为张量
		state_t = torch.tensor(np.array([obs])).to(self.device)
		# 使用移动网络计算Q值
		q_values_t = self.moving_nn(state_t)
		# 选择Q值最高的动作
		_, act_t = torch.max(q_values_t, dim=1)
		return int(act_t.item())

	# 计算损失函数
	def _calulate_loss(self, mini_batch):
		# 从mini_batch中提取状态、动作、下一个状态、奖励和终止标志
		states, actions, next_states, rewards, dones = mini_batch

		# 将数据转换为张量并移至指定设备
		states_t = torch.as_tensor(states, device=self.device)
		next_states_t = torch.as_tensor(next_states, device=self.device)
		actions_t = torch.as_tensor(actions, device=self.device)
		rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
		done_t = torch.as_tensor(dones, dtype=torch.uint8, device=self.device)

		# 计算当前状态下的Q值
		"""
		self.moving_nn(states_t): 计算所有动作的Q值
		.gather(1, actions_t[:,None]): 选择实际采取的动作的Q值
		.squeeze(-1)：去除多余的维度
		"""
		state_action_values = self.moving_nn(states_t).gather(1, actions_t[:,None]).squeeze(-1)


		"""
		计算下一个状态的Q值:
			如果使用Double DQN:
				使用移动网络选择最优动作
				使用目标网络计算该动作的Q值
			如果不使用Double DQN:
				直接使用目标网络计算下一个状态的最大Q值
		"""
		if self.double_DQN:
			double_max_action = self.moving_nn(next_states_t).max(1)[1]
			double_max_action = double_max_action.detach()
			target_output = self.target_nn(next_states_t)
			next_state_values = torch.gather(target_output, 1, double_max_action[:,None]).squeeze(-1)
		else:
			next_state_values = self.target_nn(next_states_t).max(1)[0]

		# 将下一个状态的Q值从计算图中分离,避免反向传播
		next_state_values = next_state_values.detach() 
		# 根据贝尔曼方程计算目标Q值
		expected_state_action_values = rewards_t + (self.gamma**self.n_multi_step) * next_state_values
		# 计算均方误差损失
		return nn.MSELoss()(state_action_values, expected_state_action_values)
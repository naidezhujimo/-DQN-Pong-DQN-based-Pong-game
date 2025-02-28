import numpy as np
import collections


# 回放缓存
class ReplayBuffer():
	def __init__(self, size, n_multi_step, gamma):
		"""
		size: 回放缓存的最大容量
		n_multi_step: N步DQN中的步数,用于计算多步回报
		gamma: 折扣因子
		self.buffer: 使用collections.deque初始化回放缓存,最大长度为size
		"""
		self.buffer = collections.deque(maxlen=size)
		self.n_multi_step = n_multi_step
		self.gamma = gamma

	# 获取缓冲区长度
	def __len__(self): 
		return len(self.buffer)

	# 添加经验到缓冲区
	def append(self, memory):
		self.buffer.append(memory)

	# 从缓冲区采样
	def sample(self, batch_size):
		#从回放缓存中随机采样batch_size个经验,并支持N步DQN的多步回报计算 
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)

		states = [] # 当前状态数组
		actions = [] # 动作数组
		next_states = [] # 下一个状态数组
		rewards = [] # 回报数组
		dones = [] # 终止标志数组

		# 对每个索引进行处理
		for i in indices:
			sum_reward = 0 # 用于累加N步回报
			states_look_ahead = self.buffer[i].new_obs # 初始化为当前经验的下一个状态
			done_look_ahead = self.buffer[i].done # 初始化为当前经验的终止标志

			# N步回报计算
			for n in range(self.n_multi_step):
				if len(self.buffer) > i+n:
					sum_reward += (self.gamma**n) * self.buffer[i+n].reward
					if self.buffer[i+n].done:
						states_look_ahead = self.buffer[i+n].new_obs
						done_look_ahead = True
						break
					else:
						states_look_ahead = self.buffer[i+n].new_obs
						done_look_ahead = False

			# 将处理后的数据添加到对应的列表中
			"""
			states: 当前状态
			actions: 采取的动作
			next_states: N步后的下一个状态
			rewards: N步回报
			dones: N步后的终止标志
			"""
			states.append(self.buffer[i].obs)
			actions.append(self.buffer[i].action)
			next_states.append(states_look_ahead)
			rewards.append(sum_reward)
			dones.append(done_look_ahead)

		# 将采样数据转换为numpy数组
		return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64), np.array(next_states, dtype=np.float32), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8))
import numpy as np
from collections import namedtuple

import time

from central_control import CentralControl
from buffers import ReplayBuffer


class DQNAgent():
	rewards = [] # 用于存储每局游戏的总奖励
	total_reward = 0 # 当前局游戏的总奖励
	birth_time = 0 # 智能体创建时间
	n_iter = 0 # 总迭代次数
	n_games = 0 # 总游戏局数
	ts_frame = 0 # 时间戳，用于计算FPS
	ts = time.time() # 初始化时间戳

	# 用于存储经验(观察值、动作、新观察值、奖励、终止标志)
	Memory = namedtuple('Memory', ['obs', 'action', 'new_obs', 'reward', 'done'], rename=False)


	def __init__(self, env, device, hyperparameters, summary_writer=None):
		"""
		env: 环境对象,提供观察空间和动作空间
		device: 运行设备
		hyperparameters: 超参数字典,包含折扣因子、学习率等
		summary_writer: 用于记录训练过程中的统计信息(可选)
		"""
		# 初始化CentralControl,负责管理网络的训练和更新
		self.cc = CentralControl(env.observation_space.shape, env.action_space.n, hyperparameters['gamma'], hyperparameters['n_multi_step'], hyperparameters['double_DQN'],
				hyperparameters['noisy_net'], hyperparameters['dueling'], device)

		self.cc.set_optimizer(hyperparameters['learning_rate'])

		self.birth_time = time.time() # 记录智能体的创建时间

		self.iter_update_target = hyperparameters['n_iter_update_target'] # 更新目标网络的迭代间隔
		self.buffer_start_size = hyperparameters['buffer_start_size'] # 开始训练前回放缓存需要达到的最小大小

		self.epsilon_start = hyperparameters['epsilon_start'] # 初始探索率
		self.epsilon = hyperparameters['epsilon_start'] # 当前探索率
		self.epsilon_decay = hyperparameters['epsilon_decay'] # 探索率的衰减速度
		self.epsilon_final = hyperparameters['epsilon_final'] # 最终探索率

		self.accumulated_loss = [] # 存储训练过程中的损失值
		self.device = device

		# 回放缓存，用于存储和采样经验
		self.replay_buffer = ReplayBuffer(hyperparameters['buffer_capacity'], hyperparameters['n_multi_step'], hyperparameters['gamma'])
		# 用于记录训练过程中的统计信息
		self.summary_writer = summary_writer

		# 是否使用噪声网络
		self.noisy_net = hyperparameters['noisy_net']

		# 环境对象
		self.env = env

	# 选择最优动作
	def act(self, obs):
		return self.cc.get_max_action(obs)

	# ε-贪婪策略
	def act_eps_greedy(self, obs):
		if self.noisy_net:
			return self.act(obs)
		# 以epsilon的概率随机选择动作,以1 - epsilon的概率选择最优动作
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		else:
			return self.act(obs)

	# 添加环境反馈
	def add_env_feedback(self, obs, action, new_obs, reward, done):
		# 确保 obs 和 new_obs 是 NumPy 数组
		obs_arr = np.array(obs, dtype=np.float32)
		new_obs_arr = np.array(new_obs, dtype=np.float32)

		# 检查是否含有 NaN 或 Inf
		if not np.isfinite(obs_arr).all() or not np.isfinite(new_obs_arr).all():
			print(f"NaN or Inf found in observation: {obs} or {new_obs}")

		# 将奖励值限制在[-1, 1]范围内
		reward = np.clip(reward, -1, 1)

		# 创建记忆并更新缓冲区
		new_memory = self.Memory(obs=obs_arr, action=action, new_obs=new_obs_arr, reward=reward, done=done)
		self.replay_buffer.append(new_memory)

		# 更新迭代次数、探索率和当前局游戏的总奖励
		self.n_iter += 1
		self.epsilon = max(self.epsilon_final, self.epsilon_start - self.n_iter / self.epsilon_decay)
		self.total_reward += reward

	# 采样和优化
	def sample_and_optimize(self, batch_size):

		# 如果回放缓存的大小大于buffer_start_size,则从回放缓存中采样batch_size个经验,并调用CentralControl的optimize方法进行优化
		if len(self.replay_buffer) > self.buffer_start_size:

			mini_batch = self.replay_buffer.sample(batch_size)

			l_loss = self.cc.optimize(mini_batch)
			# 将优化后的损失值添加到accumulated_loss中
			self.accumulated_loss.append(l_loss)

		# 如果当前迭代次数是iter_update_target的倍数,则更新目标网络
		if self.n_iter % self.iter_update_target == 0:
			self.cc.update_target()

	# 重置统计信息
	def reset_stats(self):
		self.rewards.append(self.total_reward) # 将当前局游戏的总奖励添加到rewards列表中
		# 重置当前局游戏的总奖励和损失列表
		self.total_reward = 0
		self.accumulated_loss = []
		self.n_games += 1 # 增加游戏局数

	# 打印信息
	def print_info(self):
		# 计算FPS（每秒帧数）
		fps = (self.n_iter - self.ts_frame) / (time.time() - self.ts)

		# 计算平均损失值
		mean_loss = np.mean(self.accumulated_loss) if self.accumulated_loss else 0.0

		mean_rew_40 = np.mean(self.rewards[-40:]) if len(self.rewards) > 40 else np.mean(self.rewards)
		mean_rew_10 = np.mean(self.rewards[-10:]) if len(self.rewards) > 10 else np.mean(self.rewards)

		print(f"{self.n_iter} {self.n_games} rew:{self.total_reward} mean_rew_40:{mean_rew_40:.2f} mean_rew_10:{mean_rew_10:.2f} eps:{self.epsilon:.2f} fps:{fps:.0f} loss:{mean_loss:.4f}")

		self.ts_frame = self.n_iter
		self.ts = time.time()
import numpy as np


def test_game(env, agent, test_episodes):
	"""
	- env: 环境对象,通常是一个强化学习环境,提供了与环境的交互接口
	- agent: 代理对象,代表强化学习中的智能体,负责根据当前状态选择动作
	- test_episodes:整数,表示要测试的回合数
	"""
	reward_games = [] # 用于存储每个测试回合的总奖励
	for _ in range(test_episodes): # 每次循环代表一个测试回合
		obs = env.reset() # 调用环境的 reset() 方法,重置环境并返回初始状态 obs
		rewards = 0 # 累计当前回合的总奖励
		while True: # 在一个回合内与环境进行交互，直到回合结束
			action = agent.act(obs) # 代理根据当前状态选择一个动作 action
			"""
			环境执行该动作并返回四个值
			- next_obs: 执行动作后的下一个状态
			- reward: 执行动作后获得的即时奖励
			- done: 一个布尔值,表示当前回合是否结束
			- _: 通常是一个包含额外信息的字典
			"""
			next_obs, reward, done, _ = env.step(action) 
			obs = next_obs 
			rewards += reward

			if done:
				reward_games.append(rewards)
				obs = env.reset()
				break

	return np.mean(reward_games) # 计算所有回合奖励的平均值
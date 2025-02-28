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

BATCH_SIZE = 64 # 每次采样的小批量大小
MAX_N_GAMES = 3000 # 最大游戏局数

ENV_NAME = "ALE/Pong-v5"  # 环境名称,这里使用的是Atari游戏Pong
SAVE_VIDEO = True # 是否保存游戏视频
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_WRITER = True # 是否使用TensorBoard记录训练过程

LOG_DIR = 'content/runs'
name = '_'.join([str(k)+'.'+str(v) for k,v in DQN_HYPERPARAMS.items()])
name = 'prv'

if __name__ == '__main__':
    # 创建环境
    env = atari_wrappers.make_env(ENV_NAME, render_mode="rgb_array")  # 传递环境对象

    if SAVE_VIDEO:
        # 使用 RecordVideo 包装器录制视频
        env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda x: x % 10 == 0)

    obs, _ = env.reset()  # 获取初始观察值和信息

    # TensorBoard 用于记录训练过程中的统计信息。
    writer = SummaryWriter(log_dir=LOG_DIR+'/'+name + str(time.time())) if SUMMARY_WRITER else None

    print('Hyperparams:', DQN_HYPERPARAMS)

    # 创建代理
    agent = DQNAgent(env, device=DEVICE, summary_writer=writer, hyperparameters=DQN_HYPERPARAMS)

    # 初始化游戏局数和迭代次数
    n_games = 0
    n_iter = 0

    # 运行 MAX_N_GAMES 次游戏
    while n_games < MAX_N_GAMES:
        action = agent.act_eps_greedy(obs) # 使用ε-贪婪策略选择动作
        # 执行动作,获取新的观察值、奖励、终止标志和截断标志
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 将经验(观察值、动作、新观察值、奖励、终止标志)添加到回放缓存中
        agent.add_env_feedback(obs, action, new_obs, reward, done)
        # 从回放缓存中采样并优化网络
        agent.sample_and_optimize(BATCH_SIZE)

        obs = new_obs # 更新当前观察值
        if done:
            n_games += 1 # 增加游戏局数
            agent.print_info() # 打印训练信息
            agent.reset_stats() # 重置智能体的统计信息
            obs, _ = env.reset()  # 重置环境并获取新的观察值和信息

    writer.close() # 关闭SummaryWriter
    env.close()  # 确保关闭环境以保存录制的视频
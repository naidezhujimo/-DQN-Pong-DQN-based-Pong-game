import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import cv2

# 类用于优化内存使用,避免重复存储相同的帧
class LazyFrames(object):
    def __init__(self, frames):
        """
        frames: 存储一系列帧
        _out: 用于缓存最终的帧数组
        """
        self._frames = frames
        self._out = None

    # 延迟计算帧数组
    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2) # 将frames中的帧沿指定轴拼接起来
            self._frames = None # 设置为None以释放内存
        return self._out

    # 转换为numpy数组
    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype) # 将数组转换为指定的数据类型
        return out

    # 返回帧数组的长度
    def __len__(self):
        return len(self._force())
    
    # 支持通过索引访问帧数组的元素
    def __getitem__(self, i):
        return self._force()[i]

# 环境包装器,用于处理那些需要按下“FIRE”按钮才能开始的游戏
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        # 断言确保环境支持“FIRE”操作
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        # 在重置环境后,执行“FIRE”操作(动作编号为1)
        obs, reward, terminated, truncated, info = self.env.step(1)
        if terminated or truncated: # 如果环境终止或截断,则重新重置环境
            self.env.reset(**kwargs)
        #再执行一次动作(编号为2),确保环境进入正常状态
        obs, reward, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info

    # 执行动作
    def step(self, ac):
        return self.env.step(ac) # 调用环境的step方法,返回完整的五元组(状态、奖励、终止标志、截断标志、信息)

# 用于跳过帧,并返回每skip帧的最大值
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # 用于存储最后两帧
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip # 跳过的帧数

    # 执行动作
    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info 

    # 重置环境
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# 将帧调整为84x84的灰度图像
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(WarpFrame, self).__init__(env)
        self.width = 84
        self.height = 84
        # 定义新的状态空间
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)
    # 处理状态值
    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # 将RGB图像转换为灰度图像
        # 将图像调整为84x84
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # 添加一个通道维度,使其成为单通道图像
        return frame[:, :, None]

# 将最后k帧堆叠在一起,形成一个新的观察空间
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k) # 使用deque存储最近的k帧
        shp = env.observation_space.shape
        # 定义堆叠后的观察空间
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype)

    # 重置环境
    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        # 在重置环境后,将初始帧重复k次
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info # 返回堆叠后的观察值和信息

    # 执行动作
    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob) # 在执行动作后,将新帧添加到frames中
        return self._get_ob(), reward, terminated, truncated, info

    # 获取堆叠后的观察值
    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames)) # 使用LazyFrames将frames中的帧堆叠起来,以优化内存使用

# 将图像的通道维度从最后移到前面,以符合PyTorch的输入格式
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        # 定义新的观察空间
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    # 处理观察值
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0) # 将通道维度从最后移到前面

# 将观察值归一化到[0, 1]范围
class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(ScaledFloatFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0 # 将观察值转换为浮点数,并除以255进行归一化

# 用于创建并包装Atari环境
def make_env(env_name, fire=True, render_mode=None):
    """
    env_name: 环境名称或已创建的环境对象
    fire: 是否启用FireResetEnv
    render_mode: 渲染模式
    """
    if isinstance(env_name, str):
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = env_name
    """
    MaxAndSkipEnv: 跳过帧并取最大值
    FireResetEnv: 在重置时按下“FIRE”按钮
    WarpFrame: 将帧调整为84x84的灰度图像
    ImageToPyTorch: 调整通道维度
    FrameStack: 堆叠4帧
    ScaledFloatFrame: 归一化观察值
    """
    env = MaxAndSkipEnv(env)
    if fire:
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ImageToPyTorch(env)
    env = FrameStack(env, 4)
    env = ScaledFloatFrame(env)
    return env
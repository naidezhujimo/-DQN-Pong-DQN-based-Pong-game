import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import cv2

# A class used to optimize memory usage by avoiding the storage of duplicate frames.
class LazyFrames(object):
    def __init__(self, frames):
        """
        frames: A list of frames to be stored.
        _out: Used to cache the final frame array.
        """
        self._frames = frames
        self._out = None

    # Delayed computation of the frame array.
    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)  # Concatenate frames along the specified axis.
            self._frames = None  # Set to None to release memory.
        return self._out

    # Convert to a NumPy array.
    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)  # Convert the array to the specified data type.
        return out

    # Return the length of the frame array.
    def __len__(self):
        return len(self._force())

    # Support accessing elements of the frame array via indexing.
    def __getitem__(self, i):
        return self._force()[i]

# An environment wrapper for handling games that require pressing the "FIRE" button to start.
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        # Assert to ensure the environment supports the "FIRE" action.
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        # After resetting the environment, perform the "FIRE" action (action number 1).
        obs, reward, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:  # If the environment terminates or truncates, reset it again.
            self.env.reset(**kwargs)
        # Perform another action (number 2) to ensure the environment enters a normal state.
        obs, reward, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info

    # Perform an action.
    def step(self, ac):
        return self.env.step(ac)  # Call the environment's step method and return the full tuple (state, reward, terminated, truncated, info).

# A wrapper used to skip frames and return the maximum value of every `skip` frames.
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # Buffer to store the last two frames.
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip  # Number of frames to skip.

    # Perform an action.
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

    # Reset the environment.
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# Resize frames to 84x84 grayscale images.
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(WarpFrame, self).__init__(env)
        self.width = 84
        self.height = 84
        # Define the new observation space.
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)

    # Process the observation value.
    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert the RGB image to grayscale.
        # Resize the image to 84x84.
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # Add a channel dimension to make it a single-channel image.
        return frame[:, :, None]

# Stack the last k frames together to form a new observation space.
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)  # Use deque to store the last k frames.
        shp = env.observation_space.shape
        # Define the stacked observation space.
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype)

    # Reset the environment.
    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        # After resetting the environment, repeat the initial frame k times.
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info  # Return the stacked observation and information.

    # Perform an action.
    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)  # After performing the action, add the new frame to frames.
        return self._get_ob(), reward, terminated, truncated, info

    # Get the stacked observation.
    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))  # Use LazyFrames to stack frames in frames to optimize memory usage.

# Move the channel dimension from the last to the front to match the PyTorch input format.
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        # Define the new observation space.
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    # Process the observation value.
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)  # Move the channel dimension from the last to the front.

# Normalize the observation values to the range [0, 1].
class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(ScaledFloatFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0  # Convert the observation to a float and normalize by dividing by 255.

# A function to create and wrap the Atari environment.
def make_env(env_name, fire=True, render_mode=None):
    """
    env_name: The name of the environment or an already created environment object.
    fire: Whether to enable FireResetEnv.
    render_mode: Rendering mode.
    """
    if isinstance(env_name, str):
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = env_name
    """
    MaxAndSkipEnv: Skip frames and take the maximum value.
    FireResetEnv: Press the "FIRE" button upon reset.
    WarpFrame: Resize frames to 84x84 grayscale images.
    ImageToPyTorch: Adjust the channel dimension.
    FrameStack: Stack 4 frames.
    ScaledFloatFrame: Normalize the observation values.
    """
    env = MaxAndSkipEnv(env)
    if fire:
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ImageToPyTorch(env)
    env = FrameStack(env, 4)
    env = ScaledFloatFrame(env)
    return env
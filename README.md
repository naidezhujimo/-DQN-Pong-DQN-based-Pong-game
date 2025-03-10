# Deep Q-Network (DQN) for Atari Games

This repository implements a Deep Q-Network (DQN) framework for training agents to play Atari games using the OpenAI Gym environment. The implementation includes various enhancements such as Double DQN, Dueling DQN, and Noisy Networks. The project is designed to be modular and easy to extend.

## Features

- **DQN Variants**: Supports standard DQN, Double DQN, Dueling DQN, and Noisy Networks.
- **Atari Wrappers**: Utilizes custom wrappers to preprocess Atari game frames (e.g., frame skipping, stacking, resizing).
- **Experience Replay**: Implements a multi-step experience replay buffer for efficient training.
- **TensorBoard Integration**: Logs training metrics to TensorBoard for visualization.
- **Epsilon-Greedy Exploration**: Supports epsilon-greedy policy with decay for exploration-exploitation trade-off.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- OpenAI Gymnasium
- Atari-Py
- NumPy
- OpenCV
- TensorBoardX

You can install the required packages using the following command:

```bash
pip install torch gymnasium atari-py numpy opencv-python tensorboardX
```

## Usage

### Training

To train the DQN agent on an Atari game (e.g., Pong), run the following command:

```bash
python main.py
```

You can customize the hyperparameters in the `main.py` file. The training process will log metrics to TensorBoard and optionally save video recordings of the agent's performance.

### Testing

To evaluate the trained agent, use the `test_game` function in `utils.py`. Example usage:

```python
from utils import test_game
from agent import DQNAgent
import gymnasium as gym

env = gym.make("ALE/Pong-v5")
agent = DQNAgent(env, device="cuda")
test_episodes = 10
average_reward = test_game(env, agent, test_episodes)
print(f"Average Reward over {test_episodes} episodes: {average_reward}")
```

## Directory Structure

- `central_control.py`: Central control module for managing the DQN networks, optimization, and target network updates.
- `main.py`: Entry point for training the DQN agent on Atari games.
- `neural_net.py`: Defines the DQN, Dueling DQN, and Noisy Linear network architectures.
- `agent.py`: Implements the DQNAgent class for interacting with the environment and managing training.
- `utils.py`: Utility functions for testing the agent's performance.
- `atari_wrappers.py`: Custom wrappers for preprocessing Atari game frames.
- `buffers.py`: Implementation of the experience replay buffer.

## Results

The agent's performance can be visualized using TensorBoard. Example metrics include:

- **Episode Rewards**: Average rewards over recent episodes.
- **Loss**: Training loss over iterations.
- **Epsilon**: Exploration rate decay.

## Contributing

Feel free to open issues or submit pull requests to improve the codebase or add new features. Contributions are welcome!

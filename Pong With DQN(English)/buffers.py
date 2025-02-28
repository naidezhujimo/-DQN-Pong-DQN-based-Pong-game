import numpy as np
import collections


# Replay Buffer
class ReplayBuffer():
    def __init__(self, size, n_multi_step, gamma):
        """
        size: The maximum capacity of the replay buffer.
        n_multi_step: The number of steps for multi-step returns in N-step DQN.
        gamma: The discount factor.
        self.buffer: Initialize the replay buffer using collections.deque with a maximum length of size.
        """
        self.buffer = collections.deque(maxlen=size)
        self.n_multi_step = n_multi_step
        self.gamma = gamma

    # Get the length of the buffer
    def __len__(self): 
        return len(self.buffer)

    # Add experience to the buffer
    def append(self, memory):
        self.buffer.append(memory)

    # Sample from the buffer
    def sample(self, batch_size):
        # Randomly sample batch_size experiences from the replay buffer and support the calculation of multi-step returns for N-step DQN.
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        states = []  # Array for current states
        actions = []  # Array for actions
        next_states = []  # Array for next states
        rewards = []  # Array for rewards
        dones = []  # Array for done flags

        # Process each index
        for i in indices:
            sum_reward = 0  # Used to accumulate the N-step return
            states_look_ahead = self.buffer[i].new_obs  # Initialize to the next state of the current experience
            done_look_ahead = self.buffer[i].done  # Initialize to the done flag of the current experience

            # N-step return calculation
            for n in range(self.n_multi_step):
                if len(self.buffer) > i + n:
                    sum_reward += (self.gamma ** n) * self.buffer[i + n].reward
                    if self.buffer[i + n].done:
                        states_look_ahead = self.buffer[i + n].new_obs
                        done_look_ahead = True
                        break
                    else:
                        states_look_ahead = self.buffer[i + n].new_obs
                        done_look_ahead = False

            # Add the processed data to the corresponding lists
            """
            states: Current state
            actions: Action taken
            next_states: Next state after N steps
            rewards: N-step return
            dones: Done flag after N steps
            """
            states.append(self.buffer[i].obs)
            actions.append(self.buffer[i].action)
            next_states.append(states_look_ahead)
            rewards.append(sum_reward)
            dones.append(done_look_ahead)

        # Convert sampled data to numpy arrays
        return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64), np.array(next_states, dtype=np.float32), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8))
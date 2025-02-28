import numpy as np

def test_game(env, agent, test_episodes):
    """
    - env: The environment object, typically a reinforcement learning environment that provides an interface for interacting with the environment.
    - agent: The agent object, representing the intelligent entity in reinforcement learning that is responsible for selecting actions based on the current state.
    - test_episodes: An integer indicating the number of episodes to test.
    """
    reward_games = []  # Used to store the total reward for each test episode.
    for _ in range(test_episodes):  # Each loop iteration represents a test episode.
        obs = env.reset()  # Call the environment's reset() method to reset the environment and return the initial state obs.
        rewards = 0  # Accumulate the total reward for the current episode.
        while True:  # Interact with the environment within an episode until it ends.
            action = agent.act(obs)  # The agent selects an action based on the current state.
            """
            The environment executes the action and returns four values:
            - next_obs: The next state after executing the action.
            - reward: The immediate reward obtained after executing the action.
            - done: A boolean indicating whether the current episode has ended.
            - _: Typically a dictionary containing additional information.
            """
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            rewards += reward

            if done:
                reward_games.append(rewards)
                obs = env.reset()
                break

    return np.mean(reward_games)  # Calculate the average reward across all episodes.
import json
from typing import Any, Dict, List

import pandas as pd
import torch

from agent import DQNAgent
from environment import TradingEnvironment
from model import DNN_QNetwork, LSTM_QNetwork
from util import EpsilonGreedyStrategy, ReplayBuffer

if __name__ == "__main__":
    num_episodes = 50
    batch_size = 32
    target_update_freq = 2  # Update the target network every 10 episodes
    epsilon_start = 1.0
    epsilon_decay = 10000
    epsilon_min = 0.001
    input_size = 20  # Number of features (OHLCV)
    hidden_size = 64  # Size of LSTM hidden state
    output_size = 3  # Number of actions (Hold, Buy, Sell)
    num_layers = 1  # Single LSTM layer

    data = pd.read_csv("BTC-USD.csv")
    window_size = 30
    price_column = "close"
    columns = data.columns

    q_network = LSTM_QNetwork(input_size, hidden_size, output_size)
    target_network = LSTM_QNetwork(input_size, hidden_size, output_size)
    target_network.load_state_dict(q_network.state_dict())

    # q_network = DNN_QNetwork(input_size, hidden_size, output_size)
    # target_network = DNN_QNetwork(input_size, hidden_size, output_size)
    # target_network.load_state_dict(q_network.state_dict())

    env = TradingEnvironment(
        data=data, window_size=window_size, columns=columns, price_column=price_column
    )
    epsilon_strategy = EpsilonGreedyStrategy(
        start=epsilon_start, end=epsilon_min, decay=epsilon_decay
    )
    replay_buffer = ReplayBuffer(capacity=10000)
    agent = DQNAgent(
        q_network=q_network,
        target_network=target_network,
        replay_buffer=replay_buffer,
        epsilon_strategy=epsilon_strategy,
        device="cpu",
    )

    logs: List[Dict[str, List[Any]]] = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False

        log: Dict[str, Any] = {
            "rewards": [],
            "actions": [],
            "total_reward": 0,
        }

        while not done:
            epsilon = epsilon_strategy.get_exploration_rate(agent.current_step)

            # Select action using epsilon-greedy strategy
            action = agent.select_action(state, epsilon)

            # Step the environment
            next_state, reward, done = env.step(action)

            # Add experience to the replay buffer
            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            agent.current_step += 1

            log["rewards"].append(reward)
            log["actions"].append(action)

            # Update the network if the replay buffer has enough samples
            if replay_buffer.size() > batch_size:
                loss = agent.update(batch_size)

        # Log the episode
        log["total_reward"] = total_reward
        logs.append(log)

        # Periodically update the target network
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Save model every 10 episodes
        if episode % 10 == 0:
            torch.save(q_network.state_dict(), f"lstm_dqn_ep_{episode}.pt")

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

    # Save the logs
    with open("logs.json", "w") as f:
        json.dump(logs, f, indent=2)

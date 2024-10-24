import json

import matplotlib.pyplot as plt
import pandas as pd


class PlotAction:
    def __init__(self, data, actions, rewards, episede, window_size=30):
        self.data = data
        self.actions = actions
        self.rewards = rewards
        self.window_size = window_size
        self.episode = episede

    def plot_signals(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data["close"], color="b", alpha=0.6)
        position = 0

        for i, action in enumerate(self.actions):
            if action == 1 and position == 0:
                position = 1
                plt.scatter(
                    self.data.index[i + self.window_size],
                    self.data["close"].iloc[i + self.window_size],
                    marker="^",
                    color="g",
                    label="Buy",
                )
            elif action == 2 and position == 1:
                position = 0
                plt.scatter(
                    self.data.index[i + self.window_size],
                    self.data["close"].iloc[i + self.window_size],
                    marker="v",
                    color="r",
                    label="Sell",
                )

        plt.title(f"Trading Actions {self.episode}")
        plt.show()


if __name__ == "__main__":
    data = pd.read_csv("BTC-USD.csv")
    with open("logs.json", "r") as f:
        logs = json.load(f)

    best_episode = -1
    # best_reward = -1e10
    # for i, episode in enumerate(logs):
    #     if episode["total_reward"] > best_reward:
    #         best_reward = episode["total_reward"]
    #         best_episode = i
    episode = best_episode
    actions = logs[episode]["actions"]
    rewards = logs[episode]["rewards"]
    plot = PlotAction(data, actions, rewards, episode)
    plot.plot_signals()

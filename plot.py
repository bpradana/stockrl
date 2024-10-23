import json

import matplotlib.pyplot as plt
import pandas as pd


class PlotAction:
    def __init__(self, data, actions, rewards):
        self.data = data
        self.actions = actions
        self.rewards = rewards

    def plot_signals(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data["close"], color="b", alpha=0.6)

        for i, action in enumerate(self.actions):
            if action == 1:
                plt.scatter(
                    self.data.index[i],
                    self.data["close"].iloc[i],
                    marker="^",
                    color="g",
                    label="Buy",
                )
            elif action == 2:
                plt.scatter(
                    self.data.index[i],
                    self.data["close"].iloc[i],
                    marker="v",
                    color="r",
                    label="Sell",
                )

        plt.title("Trading Actions")
        plt.show()


if __name__ == "__main__":
    data = pd.read_csv("BTC-USD.csv")
    with open("logs.json", "r") as f:
        logs = json.load(f)

    episode = 29
    actions = logs[episode]["actions"]
    rewards = logs[episode]["rewards"]
    plot = PlotAction(data, actions, rewards)
    plot.plot_signals()

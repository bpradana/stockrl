import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from model import LSTM_QNetwork


class Backtest:
    def __init__(self, data, initial_balance=1):
        self.data = data
        self.wallet = initial_balance
        self.share = 0
        self.total_balance = initial_balance
        self.history = []

    def run(self, actions, window_size):
        position = 0
        for i, action in enumerate(actions):
            if action == 1 and position == 0:
                position = 1
                self.share = self.wallet / self.data["close"].iloc[i + window_size]
                self.wallet = 0
                self.total_balance = (
                    self.wallet + self.share * self.data["close"].iloc[i + window_size]
                )
                self.history.append((i, "buy", self.total_balance))
            elif action == 2 and position == 1:
                position = 0
                self.wallet = self.share * self.data["close"].iloc[i + window_size]
                self.share = 0
                self.total_balance = (
                    self.wallet + self.share * self.data["close"].iloc[i + window_size]
                )
                self.history.append((i, "sell", self.total_balance))
            else:
                self.total_balance = (
                    self.wallet + self.share * self.data["close"].iloc[i + window_size]
                )
                self.history.append((i, "hold", self.total_balance))


class Predict:
    def __init__(self, model, weight):
        self.model = model
        self.weight = weight
        model.load_state_dict(torch.load(weight, weights_only=True))
        self.actions = []

    def predict(self, data, window_size):
        for i in range(window_size, len(data)):
            frame = data[i - window_size : i]
            normalized = MinMaxScaler().fit_transform(frame.values)
            with torch.no_grad():
                action = self.model(torch.tensor(normalized).float().unsqueeze(0))
            self.actions.append(action.argmax().item())

        return self.actions


class Plot:
    def __init__(self, data, history):
        self.data = data
        self.history = history

    def plot_signals(self, total_balance, window_size, show_details=False):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data["close"], color="b", alpha=0.6)

        for i, action, total_balance in self.history:
            if action == "buy":
                plt.scatter(
                    self.data.index[i + window_size],
                    self.data["close"].iloc[i + window_size],
                    marker="^",
                    color="g",
                    label="Buy",
                )
                if show_details:
                    plt.annotate(
                        f"Buy ${self.data['close'].iloc[i+window_size]}\nTotal Balance: {total_balance}",
                        (
                            self.data.index[i + window_size],
                            self.data["close"].iloc[i + window_size],
                        ),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                    )
            elif action == "sell":
                plt.scatter(
                    self.data.index[i + window_size],
                    self.data["close"].iloc[i + window_size],
                    marker="v",
                    color="r",
                    label="Sell",
                )
                if show_details:
                    plt.annotate(
                        f"Sell ${self.data['close'].iloc[i+window_size]}\nTotal Balance: {total_balance}",
                        (
                            self.data.index[i + window_size],
                            self.data["close"].iloc[i + window_size],
                        ),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                    )

        plt.title(f"Trading Actions: {total_balance}")
        plt.show()


if __name__ == "__main__":
    window_size = 30
    data = pd.read_csv("BTC-USD.csv")
    data = data[-720:]
    model = LSTM_QNetwork(20, 64, 3)
    predictor = Predict(model, "lstm_dqn_ep_50.pt")
    actions = predictor.predict(data, window_size=window_size)
    backtest = Backtest(data)
    backtest.run(actions, window_size=window_size)
    Plot(data, backtest.history).plot_signals(
        total_balance=backtest.total_balance, window_size=window_size
    )

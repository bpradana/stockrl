import numpy as np


class TradingEnvironment:
    def __init__(
        self, data, window_size, columns, price_column, transaction_cost=0.001
    ):
        self.data = data  # The normalized OHLCV data
        self.window_size = window_size  # Number of days in each state
        self.transaction_cost = transaction_cost  # Fee for each trade
        self.current_step = window_size  # Start after the first full window
        self.done = False  # Whether the episode is finished
        self.position = 0  # 1 for holding a long position, 0 for no position
        self.cash = 1.0  # Start with an initial portfolio value (e.g., $1)
        self.holdings = 0  # Number of Bitcoin held
        self.starting_portfolio_value = (
            1.0  # Initial portfolio value for profit calculation
        )
        self.portfolio_value = self.cash  # Portfolio value (cash + holdings)
        self.columns = columns
        self.price_column = price_column

    def _next_observation(self):
        """Get the window of data."""
        frame = self.data[self.current_step - self.window_size : self.current_step]

        obs = []
        for column in self.columns:
            normalized = (frame[column] - frame[column].mean()) / frame[column].std()
            obs.append(normalized.values)

        return np.array(obs)

    def reset(self):
        """Reset the environment for a new episode."""
        self.current_step = self.window_size
        self.done = False
        self.position = 0
        self.cash = 1.0
        self.holdings = 0
        self.portfolio_value = self.cash
        return self._next_observation()

    def get_portfolio_value(self):
        """Calculate the portfolio value (cash + holdings value)."""
        current_price = self.data.iloc[self.current_step][
            self.price_column
        ]  # 'Close' price
        return self.cash + self.holdings * current_price

    def step(self, action):
        """
        Take a step in the environment.
        Actions: 0 = Hold, 1 = Buy, 2 = Sell
        """
        current_price = self.data.iloc[self.current_step][
            self.price_column
        ]  # 'Close' price

        # Reward based on portfolio value change
        previous_portfolio_value = self.portfolio_value

        # Action 1: Buy
        if action == 1 and self.position == 0:
            # Buy Bitcoin
            self.holdings = self.cash / current_price
            self.cash = 0  # All cash is now in Bitcoin
            self.position = 1  # Holding a long position
            self.portfolio_value = self.get_portfolio_value()
            reward = (
                self.portfolio_value - previous_portfolio_value - self.transaction_cost
            )

        # Action 2: Sell
        elif action == 2 and self.position == 1:
            # Sell Bitcoin
            self.cash = self.holdings * current_price
            self.holdings = 0  # Sold all Bitcoin
            self.position = 0  # No position
            self.portfolio_value = self.get_portfolio_value()
            reward = (
                self.portfolio_value - previous_portfolio_value - self.transaction_cost
            )

        # Action 0: Hold
        else:
            # No change, just update portfolio value
            self.portfolio_value = self.get_portfolio_value()
            reward = self.portfolio_value - previous_portfolio_value

        # Move to the next time step
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True  # End the episode if we're out of data

        # Return the new state, reward, and done flag
        next_state = self._next_observation()
        return next_state, reward, self.done

import yfinance as yf
from ta.momentum import rsi
from ta.trend import MACD, ema_indicator, sma_indicator
from ta.volatility import BollingerBands
from ta.volume import on_balance_volume


class FeatureEngineering:
    def __init__(self, data):
        self.data = data

    def _add_sma(self, period):
        self.data[f"sma_{period}"] = sma_indicator(self.data["close"], window=period)

    def _add_ema(self, period):
        self.data[f"ema_{period}"] = ema_indicator(self.data["close"], window=period)

    def _add_rsi(self, period):
        self.data[f"rsi_{period}"] = rsi(self.data["close"], window=period)

    def _add_bb(self, period):
        bb = BollingerBands(self.data["close"], window=period)
        self.data[f"bb_hband_{period}"] = bb.bollinger_hband()
        self.data[f"bb_lband_{period}"] = bb.bollinger_lband()
        self.data[f"bb_mavg_{period}"] = bb.bollinger_mavg()

    def _add_macd(self):
        macd = MACD(self.data["close"])
        self.data["macd"] = macd.macd()
        self.data["macd_signal"] = macd.macd_signal()
        self.data["macd_diff"] = macd.macd_diff()

    def _add_obv(self):
        self.data["obv"] = on_balance_volume(self.data["close"], self.data["volume"])

    def add_indicators(self):
        # Add SMA
        self._add_sma(20)
        self._add_sma(50)
        self._add_sma(200)

        # Add EMA
        self._add_ema(12)
        self._add_ema(26)
        self._add_ema(50)

        # Add RSI
        self._add_rsi(14)

        # Add Bollinger Bands
        self._add_bb(20)

        # Add MACD
        self._add_macd()

        # Add OBV
        self._add_obv()

    def get_data(self):
        return self.data


if __name__ == "__main__":
    data = yf.download("BTC-USD", start="2015-01-01", end="2024-10-01")
    data = data.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    fe = FeatureEngineering(data)
    fe.add_indicators()
    data = fe.get_data()
    data = data.dropna()
    data.to_csv("BTC-USD.csv", index=False)

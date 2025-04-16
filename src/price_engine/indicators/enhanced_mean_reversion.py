import numpy as np

class EnhancedMeanReversion:
    def __init__(self, window=20):
        self.window = window
        self.hold_counter = 0
        self.in_trade = False

    def auto_threshold(self, volatility):
        if volatility < 1:
            return 1.2  # very stable
        elif volatility < 5:
            return 1.5  # moderate volatility
        else:
            return 2.0  # high volatility

    def auto_min_hold_days(self, volatility):
        if volatility < 1:
            return 5
        elif volatility < 5:
            return 3
        else:
            return 2

    def auto_allow_short(self, asset_name):
        # Allow shorting for stocks or futures, not crypto for now
        return not asset_name.lower().endswith("usdt")

    def decide(self, data_window, current_position, asset_name="unknown"):
        if len(data_window) < self.window:
            return None

        prices = [p['price'] for p in data_window[-self.window:]]
        latest_price = data_window[-1]['price']
        mean_price = np.mean(prices)
        std_dev = np.std(prices)

        if std_dev == 0:
            return None

        z_score = (latest_price - mean_price) / std_dev
        threshold = self.auto_threshold(std_dev)
        min_hold_days = self.auto_min_hold_days(std_dev)
        allow_short = self.auto_allow_short(asset_name)

        print(f"[Auto MR] Asset: {asset_name} | Price: {latest_price:.2f}, Z: {z_score:.2f}, Thresh: {threshold}, Pos: {current_position}, Hold: {self.hold_counter}")

        if self.hold_counter < min_hold_days:
            self.hold_counter += 1
            return None

        # Long entry
        if z_score < -threshold and current_position != "long":
            self.hold_counter = 0
            return "buy"

        # Long exit
        elif z_score > threshold and current_position == "long":
            self.hold_counter = 0
            return "sell"

        # Short entry
        elif allow_short and z_score > threshold and current_position != "short":
            self.hold_counter = 0
            return "short"

        # Short exit
        elif allow_short and z_score < -threshold and current_position == "short":
            self.hold_counter = 0
            return "cover"

        return None

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from price_engine.indicators.bollinger_bands import BollingerBands
from price_engine.indicators.mean_reversion import MeanReversion

def plot_live_price(symbols, asset_type):
    indicators = {symbol: {
        'bb': BollingerBands(window=20, num_std=2),
        'mr': MeanReversion(window=20, threshold=2.0)
    } for symbol in symbols}

    fig, axes = plt.subplots(len(symbols), 1, figsize=(12, 6 * len(symbols)), sharex=True)
    if len(symbols) == 1:
        axes = [axes]  # Ensure iterable

    def animate(i):
        for idx, symbol in enumerate(symbols):
            filename = f"{symbol.lower()}_price_log.csv"
            ax = axes[idx]
            ax.cla()  # Clear previous frame

            if not os.path.exists(filename):
                print(f"CSV for {symbol} not found.")
                continue

            try:
                data = pd.read_csv(filename, header=None, names=['Timestamp', 'Price'])
                data['Timestamp'] = pd.to_datetime(data['Timestamp'])
                data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

                price_data = [{"price": p} for p in data['Price'].tolist()]

                # Calculate Indicators
                bb = indicators[symbol]['bb'].calculate(price_data)
                mr = indicators[symbol]['mr'].calculate(price_data)

                # Plot Price and Bands
                ax.plot(data['Timestamp'], data['Price'], label=f"{symbol.upper()} Price", color='dodgerblue')
                ax.axhline(bb['upper_band'], color='red', linestyle='--', label='Upper Band')
                ax.axhline(bb['lower_band'], color='green', linestyle='--', label='Lower Band')
                ax.axhline(bb['moving_avg'], color='gray', linestyle='--', label='Moving Avg')

                # Show mean reversion signals
                if mr['overbought']:
                    ax.set_title(f"{symbol.upper()} - ⚠️ Overbought")
                elif mr['oversold']:
                    ax.set_title(f"{symbol.upper()} - 📉 Oversold")
                else:
                    ax.set_title(f"{symbol.upper()}")

                ax.legend(loc='upper left')
                ax.set_ylabel("Price (USDT)")
                ax.tick_params(axis='x', rotation=45)

            except Exception as e:
                print(f"Error animating {symbol}: {e}")

        plt.tight_layout()

    ani = FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)
    plt.show()

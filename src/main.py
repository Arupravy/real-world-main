# src/main.py
import time 
import argparse
import asyncio
from colorama import Fore, Style
from datetime import datetime
from tabulate import tabulate
import plotly.graph_objs as go


from price_engine.aggregator import PriceAggregator
from price_engine.price_calculator import PriceCalculator
from price_engine.indicators.bollinger_bands import BollingerBands
from price_engine.indicators.mean_reversion import MeanReversion
from price_engine.api_handler import fetch_price_from_api
from price_engine.data_sources.websocket_handler import BinanceWebSocketClient
from price_engine.price_stream_to_csv import stream_prices_to_csv
from price_engine.live_price_plot import plot_live_price

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fetch and display asset prices.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "api-live", "ws-live", "historical", "stream-to-csv", "live-plot"],
        required=True,
         help="Mode to run: 'live', 'api-live', 'ws-live', 'historical', 'stream-to-csv', or 'live-plot'.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        nargs='+',
        required=True,
        help="Symbol of the asset. One or more trading symbols (e.g., BTCUSDT, ETHUSDT, AAPL).",
    )
    parser.add_argument(
        "--asset-type",
        type=str,
        choices=["crypto", "stock"],
        required=True,
        help="Type of asset: 'crypto' or 'stock'.",
    )
    parser.add_argument(
        "--from",
        dest="from_date",
        type=str,
        help="Start date for historical prices (format: YYYY-MM-DD). Required for historical mode.",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        type=str,
        help="End date for historical prices (format: YYYY-MM-DD). Required for historical mode.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Window size for indicators (default: 20). Minimum 5.",
    )
    parser.add_argument(
        "--std-dev",
        type=float,
        default=2.0,
        dest="std_dev",
        help="Number of standard deviations for Bollinger Bands (default: 2.0).",
    )
    parser.add_argument(
    "--plot",
    action="store_true",
    help="Show interactive price trend plot (only for historical mode).",
    )
    return parser.parse_args()

def run_live_mode(aggregator, symbols: list[str], window: int, std_dev: float):
    """Fetch and display live prices with indicators."""
    for symbol in symbols:
        # Fetch prices and calculate weighted average
        prices = aggregator.get_all_prices(symbol)
        if not prices:
            print(f"No prices available for symbol {symbol}")
            continue

        prices = PriceCalculator.handle_outliers(prices)
        weights = {source: info["weight"] for source, info in aggregator.sources.items()}
        weighted_avg_price = PriceCalculator.calculate_weighted_average(prices, weights)

        # Display current prices
        print(f"\nLive Prices for {symbol} from all sources:")
        for source, price in prices.items():
            print(f"{source}: {price}")
        print(f"\nWeighted Average Price: {weighted_avg_price}")

        # Get price history for indicators
        price_history = aggregator.get_price_history()
        relevant_prices = [entry["price"] for entry in price_history if entry["symbol"] == symbol]

        # Calculate indicators if we have enough data
        if len(relevant_prices) >= window:
            price_data = [{"price": p} for p in relevant_prices[-window:]]

            bb = BollingerBands(window=window, num_std=std_dev)
            mr = MeanReversion(window=window, threshold=std_dev)

            bb_result = bb.calculate(price_data)
            mr_result = mr.calculate(price_data)

            print(f"\nTechnical Indicators ({window}-period):")
            print("Bollinger Bands:")
            print(f"  MA: {bb_result['moving_avg']:.2f}")
            print(f"  Upper: {bb_result['upper_band']:.2f}")
            print(f"  Lower: {bb_result['lower_band']:.2f}")

            print("\nMean Reversion:")
            status = "Overbought" if mr_result['overbought'] else "Oversold" if mr_result['oversold'] else "Neutral"
            print(f"  Status: {status}")
        else:
            print(f"\nNeed {window - len(relevant_prices)} more price points for indicators")


def run_stream_to_csv_mode(symbols, asset_type):
    print(f"💾 Starting CSV stream for {', '.join(symbols)} ({asset_type})...")
    try:
        stream_prices_to_csv(symbols=symbols, asset_type=asset_type)
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Stopped CSV streaming.{Style.RESET_ALL}")

def run_live_plot_mode(symbols, asset_type):
    print(f"📈 Starting live plot for {', '.join(symbols)} ({asset_type})...")
    try:
        plot_live_price(symbols=symbol_list, asset_type=args.asset_type)
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Stopped live plotting.{Style.RESET_ALL}")

def run_historical_mode(aggregator, symbol: str, from_date: str, to_date: str, window: int, std_dev: float, args):
    """Fetch and display historical prices with indicators."""
    # Get historical prices
    historical_prices = aggregator.get_historical_prices(symbol, from_date, to_date)
    
    if not historical_prices:
        print(f"\nNo historical prices available for {symbol} in the specified date range.")
        return

    # Display historical prices in a table
    print(f"\nHistorical Prices for {symbol}:")

    # Identify the most recent and previous entries
    latest_entry = historical_prices[-1]
    prev_entry = historical_prices[-2] if len(historical_prices) >= 2 else None

    # Determine price movement
    arrow = ""
    if prev_entry:
        if latest_entry["price"] > prev_entry["price"]:
            arrow = f"{Fore.GREEN}⬆️{Style.RESET_ALL}"
        elif latest_entry["price"] < prev_entry["price"]:
            arrow = f"{Fore.RED}⬇️{Style.RESET_ALL}"
        else:
            arrow = "➡️"

    # Build the table rows manually
    table_rows = []
    for entry in historical_prices:
        date = entry["date"]
        price = entry["price"]
        if entry == latest_entry:
            colored_date = f"{Fore.GREEN}{date}{Style.RESET_ALL}"
            colored_price = f"{Fore.GREEN}{price:.2f} {arrow}{Style.RESET_ALL}"
            table_rows.append([colored_date, colored_price])
        else:
            table_rows.append([date, f"{price:.2f}"])

    print(tabulate(table_rows, headers=["date", "price"], tablefmt="pretty"))

    # Prepare data for indicators
    prices = [entry["price"] for entry in historical_prices]
    
    # Calculate indicators if we have enough data
    if len(prices) >= window:
        price_data = [{"price": p} for p in prices[-window:]]
        
        bb = BollingerBands(window=window, num_std=std_dev)
        mr = MeanReversion(window=window, threshold=std_dev)
        
        bb_result = bb.calculate(price_data)
        mr_result = mr.calculate(price_data)

        print(f"\nTechnical Indicators ({window}-period):")
        print("Bollinger Bands:")
        print(f"  MA: {bb_result['moving_avg']:.2f}")
        print(f"  Upper: {bb_result['upper_band']:.2f}")
        print(f"  Lower: {bb_result['lower_band']:.2f}")
        
        print("\nMean Reversion:")
        status = "Overbought" if mr_result['overbought'] else "Oversold" if mr_result['oversold'] else "Neutral"
        print(f"  Status: {status}")
    else:
        print(f"\nNot enough data for {window}-day indicators")
    
    # Optional interactive plotting with Plotly
    if args.plot:
        dates = [entry["date"] for entry in historical_prices]
        prices = [entry["price"] for entry in historical_prices]

        fig = go.Figure()

        # Main price line
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines+markers',
            name='Price',
            line=dict(color='royalblue', width=2)
        ))

        # Add Bollinger Bands (if indicators were calculated)
        if len(prices) >= window:
            bb_prices = prices[-window:]
            ma = bb_result['moving_avg']
            upper = bb_result['upper_band']
            lower = bb_result['lower_band']

            fig.add_trace(go.Scatter(
                x=dates[-window:],
                y=[upper] * window,
                mode='lines',
                name='Upper Band',
                line=dict(color='red', dash='dash')
            ))

            fig.add_trace(go.Scatter(
                x=dates[-window:],
                y=[ma] * window,
                mode='lines',
                name='Moving Avg',
                line=dict(color='green', dash='dot')
            ))

            fig.add_trace(go.Scatter(
                x=dates[-window:],
                y=[lower] * window,
                mode='lines',
                name='Lower Band',
                line=dict(color='red', dash='dash')
            ))

            fig.update_layout(
            title=f"{symbol} Price Trend ({from_date} to {to_date})",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified"
            )

            fig.show()

def run_api_live_mode(symbol: str, asset_type: str):
    """Fetch live price using API handler and print every 5 seconds."""
    print(f"🔄 Starting API polling for {symbol} ({asset_type})...\n")
    try:
        while True:
            price = fetch_price_from_api(symbol, asset_type)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if price is not None:
                print(f"{Fore.CYAN}[{now}]{Style.RESET_ALL} {symbol}: {Fore.YELLOW}{price:.2f}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to fetch price for {symbol}.{Style.RESET_ALL}")
            time.sleep(5)  # Sleep is safe even in sync loop here
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Stopped API live mode.{Style.RESET_ALL}")

def run_websocket_live_mode(symbols):
    """Run real-time streaming using WebSocket."""
    print(f"🛰️ Subscribing to {', '.join(symbols).upper()} WebSocket stream...\n")
    aggregator = PriceAggregator(asset_type="crypto", symbols=symbols)
    client = aggregator.sources["binance_ws"]["handler"]
    client.start()


if __name__ == "__main__":
    args = parse_args()
    if isinstance(args.symbol, list):
        symbol_list = [s.strip().upper() for s in args.symbol]
    else:
        args.symbol = args.symbol.upper()

    # Create a normalized list    
    symbol_list = args.symbol if isinstance(args.symbol, list) else [args.symbol]

    # Validate window size
    if args.window < 5:
        print("Warning: Window size too small. Setting to minimum value of 5.")
        args.window = 5

    # Initialize the appropriate aggregator
    aggregator = PriceAggregator(asset_type=args.asset_type, symbols=symbol_list)

    if args.mode == "live":
        run_live_mode(aggregator, [s.upper() for s in args.symbol], window=args.window, std_dev=args.std_dev)

    elif args.mode == "api-live":
        run_api_live_mode(args.symbol, args.asset_type)

    elif args.mode == "ws-live":
        run_websocket_live_mode(symbol_list)
    
    elif args.mode == "stream-to-csv":
        run_stream_to_csv_mode(symbol_list, args.asset_type)
    
    elif args.mode == "live-plot":
        run_live_plot_mode(symbol_list, args.asset_type)

    elif args.mode == "historical":
        if not args.from_date or not args.to_date:
            print("Error: --from and --to dates are required for historical mode.")
        else:
            for symbol in args.symbol:
                 run_historical_mode(aggregator, symbol=symbol, from_date=args.from_date, to_date=args.to_date,
                        window=args.window, std_dev=args.std_dev, args=args)
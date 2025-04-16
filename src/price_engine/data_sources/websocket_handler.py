# C:\real-world-main\src\price_engine\data_sources\websocket_handler.py
import websocket
import threading
import json
import time
from datetime import datetime
from colorama import init, Fore

init(autoreset=True)

class BinanceWebSocketClient:
    def __init__(self, symbols, on_price_update=None):
        self.symbols = symbols
        self.previous_prices = {}
        self.on_price_update = on_price_update  # 💥 You missed this line earlier

    def on_message(self, ws, message):
        data = json.loads(message)
        symbol = data['s']
        price = float(data['p'])

        now = datetime.now().strftime('%H:%M:%S')

        # Update price history
        if symbol not in self.previous_prices:
            self.previous_prices[symbol] = price
            diff_percent = 0
        else:
            old_price = self.previous_prices[symbol]
            diff_percent = ((price - old_price) / old_price) * 100
            self.previous_prices[symbol] = price

        # 🔄 Real-time strategy logic
        if self.on_price_update:
            self.on_price_update(symbol, price)  # forward price to strategy
        else:
            # 🖨️ Fancy print if no trading logic
            if diff_percent > 0:
                color = Fore.GREEN
                change = f"🔺 +{diff_percent:.2f}%"
            elif diff_percent < 0:
                color = Fore.RED
                change = f"🔻 {diff_percent:.2f}%"
            else:
                color = Fore.YELLOW
                change = "⏸️  0.00%"

            print(f"{color}[{now}] {symbol}: {price:.2f} {change}")

    def on_error(self, ws, error):
        print(Fore.RED + f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(Fore.LIGHTBLACK_EX + "WebSocket closed.")

    def on_open(self, ws):
        print(Fore.CYAN + "WebSocket connection opened.")

    def create_ws(self, symbol):
        stream_symbol = symbol.lower()
        url = f"wss://stream.binance.com:9443/ws/{stream_symbol}@trade"
        ws = websocket.WebSocketApp(
            url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        thread = threading.Thread(target=ws.run_forever)
        thread.daemon = True
        thread.start()

    def start(self):
        for symbol in self.symbols:
            self.create_ws(symbol)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(Fore.LIGHTBLUE_EX + "\nStreaming stopped by user.")

# 👇 Wrapper function you can import
def start_price_feed(symbols, on_price_update):
    client = BinanceWebSocketClient(symbols, on_price_update)
    client.start()
import csv
import datetime
import websocket
import threading
import json
import time


def stream_single_symbol(symbol):
    def write_to_csv(price):
        filename = f"{symbol}_price_log.csv"
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([now, price])

    def on_message(ws, message):
        data = json.loads(message)
        price = data['p']
        print(f"{symbol.upper()} price: {price}")
        write_to_csv(price)

    def on_error(ws, error):
        print(f"Error for {symbol.upper()}: {error}")

    def on_close(ws, close_status_code, close_msg):
        print(f"### Closed WebSocket for {symbol.upper()} ###")

    def on_open(ws):
        print(f"✅ Connected to Binance WebSocket for {symbol.upper()}")

    ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                on_open=on_open)
    ws.run_forever()


def stream_prices_to_csv(symbols, asset_type="crypto"):
    print(f"💾 Starting CSV stream for {', '.join(symbols)} ({asset_type})...\n")
    threads = []

    for symbol in symbols:
        t = threading.Thread(target=stream_single_symbol, args=(symbol,))
        t.daemon = True  # <-- allow threads to exit when main thread ends
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(1)  # keep main thread alive
    except KeyboardInterrupt:
        print("\n⛔️ Stream interrupted by user. Exiting...")

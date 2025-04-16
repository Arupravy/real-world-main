# Price Engine

A Python-based price engine that aggregates cryptocurrency prices from multiple sources (e.g., Binance, CoinGecko, Coinbase) and provides live and historical price data.

## Features
- Fetch live prices from multiple sources.
- Fetch historical prices for a specified date range.
- Calculate weighted average prices.
- Display prices in a well-formatted table.

## Usage

### Live Mode
Fetch and display live prices:
```bash
python src/main.py --mode live
```


### Historical Mode
Fetch and display historical prices for a date range:

```bash
python src/main.py --mode historical --from 2025-03-19 --to 2025-03-20

```
## Installation

### Clone the repository:

```bash
git clone https://github.com/arupravy/price-engine.git

```

### Install dependencies:

```bash
pip install -r requirements.txt
```
## Contributing


Contributions are welcome! Please open an issue or submit a pull request.
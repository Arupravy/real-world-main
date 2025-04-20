python src/main.py --mode historical --symbol AAPL MSFT TSLA --asset-type stock --from 2024-01-01 --to 2024-01-10

python src/main.py --mode live --symbol AAPL TSLA --asset-type stock

python src/main.py --mode historical --symbol BTCUSDT ETHUSDT --asset-type crypto --from 2024-01-01 --to 2024-01-10

python src/main.py --mode ws-live --symbol BTCUSDT --asset-type crypto

python src/main.py --mode ws-live --symbol BTCUSDT ETHUSDT --asset-type crypto

python src/main.py --mode api-live --symbol ETHUSDT --asset-type crypto



python src/backtesting_engine/backtest_runner.py --symbols AAPL --allocations 100 --start 2020-01-01 --end 2024-01-01 --strategy mean_reversion --asset_type stock

python src/backtesting_engine/backtest_runner.py --symbols TSLA,AAPL,MSFT --allocations 40,30,30 --start 2020-01-01 --end 2024-01-01 --strategy mean_reversion --asset_type stock



python src/backtesting_engine/backtest_runner.py --symbols NVDA,^GSPC,INTC --allocations 40,30,30 --start 2019-01-01 --end 2021-01-01 --strategy mean_reversion --asset_type stock

PS C:\real-world-main> python src/backtesting_engine/backtest_runner.py --symbols BTCUSDT,ETHUSDT --allocations 60,40 --start 2020-01-01 --end 2025-01-01 --strategy mean_reversion --asset_type crypto





python src/main.py --mode live-plot --symbol BTCUSDT ETHUSDT DOGEUSDT --asset-type crypto


python src/main.py --mode stream-to-csv --symbol BTCUSDT ETHUSDT DOGEUSDT --asset-type crypto


streamlit run src/streamlit_dashboard.py
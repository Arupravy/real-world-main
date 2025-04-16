import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import threading
import time
import smtplib
import kaleido
import plotly
from email.message import EmailMessage
from io import StringIO

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import plotly.io as pio
import io

from backtesting_engine.real_time_runner import RealTimeTrader
from price_engine.data_sources.websocket_handler import start_price_feed




SESSION_HISTORY_FILE = "session_history.json"

def load_history():
    if os.path.exists(SESSION_HISTORY_FILE):
        with open(SESSION_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(SESSION_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)



# --- Fixed Email utility function ---
def send_email_with_chart(summary, logs, fig, recipient="divyanshuydv0002@gmail.com"):
    sender_email = "alert.realworld@gmail.com"
    subject = "📈 Final Trading PnL Report"
    
    body = f"""
📈 Final Trading Summary:
-------------------------
• Final Portfolio Value: ${summary['final_portfolio_value']:.2f}
• Final PnL: ${summary['final_pnl']:.2f}
• Cash Balance: ${summary['cash_balance']:.2f}
• Unrealized PnL: ${summary['unrealized_pnl']:.2f}

🧾 Recent Trade Logs:
---------------------
{chr(10).join(logs[-10:])}
"""

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        # Convert plotly figure to image
        img_bytes = pio.to_image(fig, format='jpg', engine='kaleido')
        image = MIMEImage(img_bytes, name="pnl_chart.jpg")
        msg.attach(image)
        
        # Send email with timeout
        with smtplib.SMTP("smtp.sendgrid.net", 587, timeout=10) as server:
            server.starttls()
            server.login("apikey", "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")
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email error: {str(e)}")
        return False


# --- Streamlit Trading Dashboard ---

# Shared state
if "trader" not in st.session_state:
    st.session_state.trader = None

if "runner_thread" not in st.session_state:
    st.session_state.runner_thread = None

if "last_summary" not in st.session_state:
    st.session_state.last_summary = {}

if "last_logs" not in st.session_state:
    st.session_state.last_logs = []

if "last_timeline" not in st.session_state:
    st.session_state.last_timeline = []

if "show_summary" not in st.session_state:
    st.session_state.show_summary = False

if "completed_runs" not in st.session_state:
    st.session_state.completed_runs = load_history()

# Sidebar config
st.sidebar.title("⚙️ Trading Config")
initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=10000)
runtime = st.sidebar.slider("Runtime (seconds)", 60, 1800, 300, step=60)
symbols = st.sidebar.multiselect("Symbols to Track", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"], default=["BTCUSDT", "ETHUSDT"])

# Start button
if st.sidebar.button("▶️ Start Trading") and st.session_state.trader is None:
    st.session_state.trader = RealTimeTrader(capital=initial_capital, runtime=runtime)
    st.session_state.runner_thread = threading.Thread(
        target=start_price_feed,
        args=(symbols, st.session_state.trader.on_price_update),
        daemon=True
    )
    st.session_state.runner_thread.start()
    st.success("🚀 Trading session started!")

# Stop button
if st.sidebar.button("⏹ Stop Trading"):
    if st.session_state.trader:
        st.session_state.trader.stop()
        st.session_state.last_summary = st.session_state.trader.get_summary()
        st.session_state.last_logs = st.session_state.trader.logs.copy()
        st.session_state.last_timeline = st.session_state.trader.get_timeline().copy()

        session_info = {
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Symbols": ', '.join(symbols),
            "Initial Capital": initial_capital,
            "Final Portfolio Value": st.session_state.last_summary["final_portfolio_value"],
            "PnL": st.session_state.last_summary["final_pnl"],
            "Duration (s)": runtime
        }

        st.session_state.completed_runs.append(session_info)
        save_history(st.session_state.completed_runs)
        st.session_state.trader = None
        st.session_state.show_summary = True
        st.success("✅ Trading stopped.")

# Handle summary display after session ends
if st.session_state.trader is None:
    if st.session_state.get("show_summary"):
        st.success("🎉 Trading session completed!")

        st.subheader("📋 Final Summary")
        final_pnl = st.session_state.last_summary.get("final_pnl", 0)
        final_value = st.session_state.last_summary.get("final_portfolio_value", 0)

        st.metric("💰 Final Portfolio Value", f"${final_value:,.2f}")
        st.metric("📊 Final PnL", f"${final_pnl:,.2f}")

        # Generate chart from saved timeline
        pnl_df = pd.DataFrame(st.session_state.last_timeline)
        fig = go.Figure()
        if not pnl_df.empty and {"timestamp", "portfolio_value"}.issubset(pnl_df.columns):
            pnl_df["timestamp"] = pd.to_datetime(pnl_df["timestamp"])
            fig.add_trace(go.Scatter(x=pnl_df["timestamp"], y=pnl_df["portfolio_value"],
                                     mode='lines+markers', line=dict(color="skyblue"),
                                     name="Portfolio Value"))
            fig.update_layout(
                            title="📈 Portfolio Value Over Time",
                            xaxis_title="Timestamp",
                            yaxis_title="Portfolio ($)",
                            template="plotly_dark",
                            plot_bgcolor="#0e1117",
                            paper_bgcolor="#0e1117",
                            font=dict(color="#ffffff"),
                            height=450
                        )

        if st.button("📧 Send Final Email Summary"):
            if send_email_with_chart(st.session_state.last_summary, 
                                   st.session_state.last_logs, 
                                   fig):
                st.success("✅ Email sent successfully!")
            else:
                st.error("❌ Failed to send email")
        
        # Display Session History Table
        st.write("DEBUG - completed_runs:", st.session_state.get("completed_runs", "Not Found"))

        if st.session_state.completed_runs:
            st.subheader("📊 Session History")
            history_df = pd.DataFrame(st.session_state.completed_runs)
            st.dataframe(history_df)

        st.stop()
    else:
        st.info("Configure and start the trading bot from the sidebar.")
        st.stop()

# Active session continues here
trader = st.session_state.trader

elapsed = int(time.time() - trader.start_time)
remaining = max(runtime - elapsed, 0)
st.sidebar.metric("⏳ Time Remaining", f"{remaining} sec")

# Auto-close if trader ends silently
if not trader.is_active:
    st.session_state.trader = None
    st.session_state.show_summary = True
    st.rerun()

# --- Main UI ---
st.title("📊 Real-Time Trading Dashboard")

# Portfolio Summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💼 Initial Capital", f"${trader.initial_capital:,.2f}")
with col2:
    st.metric("💵 Cash Balance", f"${trader.cash_balance:,.2f}")
with col3:
    unrealized = trader.calculate_unrealized_pnl()
    st.metric("📈 Unrealized PnL", f"${unrealized:,.2f}")
with col4:
    portfolio_value = trader.cash_balance + unrealized
    st.metric("📊 Portfolio Value", f"${portfolio_value:,.2f}")

# PnL Chart
expected_keys = {"timestamp", "portfolio_value"}
cleaned_timeline = [
    entry for entry in trader.pnl_timeline
    if isinstance(entry, dict) and expected_keys.issubset(entry)
]
pnl_df = pd.DataFrame(cleaned_timeline)
fig = go.Figure()

if not pnl_df.empty:
    pnl_df["timestamp"] = pd.to_datetime(pnl_df["timestamp"])
    fig.add_trace(go.Scatter(x=pnl_df["timestamp"], y=pnl_df["portfolio_value"],
                             mode='lines+markers', line=dict(color="skyblue"),
                             name="Portfolio Value"))
    fig.update_layout(
                        title="📈 Portfolio Value Over Time",
                        xaxis_title="Timestamp",
                        yaxis_title="Portfolio ($)",
                        template="plotly_dark",
                        plot_bgcolor="#0e1117",
                        paper_bgcolor="#0e1117",
                        font=dict(color="#ffffff"),
                        height=450
                    )
st.plotly_chart(fig, use_container_width=True)

# Open Positions
if trader.positions:
    st.subheader("📌 Open Positions")
    pos_data = []
    for symbol, pos in trader.positions.items():
        current_price = trader.data[symbol][-1]["price"] if trader.data[symbol] and "price" in trader.data[symbol][-1] else 0
        pnl = (current_price - pos["entry_price"]) * pos["size"]
        pos_data.append({
            "Symbol": symbol,
            "Side": pos["side"].upper(),
            "Entry Price": round(pos["entry_price"], 2),
            "Size": round(pos["size"], 4),
            "Current Price": round(current_price, 2),
            "PnL": round(pnl, 2)
        })
    st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
else:
    st.info("No open positions currently.")

# Trade Logs
st.subheader("🧾 Trade Logs")
if trader.logs:
    st.code("\n".join(trader.logs[-20:]), language="bash")
else:
    st.write("Waiting for trade signals...")
    

# Download logs
if st.button("📥 Download Logs as CSV"):
    csv_buffer = StringIO()
    log_df = pd.DataFrame(trader.logs, columns=["Trade Logs"])
    log_df.to_csv(csv_buffer, index=False)
    st.download_button("Download Logs", csv_buffer.getvalue(), file_name="trade_logs.csv", mime="text/csv")


# Manual email send during session
if st.button("📧 Send Email Summary"):
    summary = {
        "final_portfolio_value": portfolio_value,
        "final_pnl": portfolio_value - trader.initial_capital,
        "cash_balance": trader.cash_balance,
        "unrealized_pnl": unrealized
    }
    if send_email_with_chart(summary, trader.logs, fig):
        st.success("✅ Email sent successfully!")
    else:
        st.error("❌ Failed to send email")

# Save state
st.session_state.last_summary = {
    "final_portfolio_value": portfolio_value,
    "final_pnl": portfolio_value - trader.initial_capital,
    "cash_balance": trader.cash_balance,
    "unrealized_pnl": unrealized
}
st.session_state.last_logs = trader.logs.copy()
st.session_state.last_timeline = trader.pnl_timeline.copy()

# Reset
if st.sidebar.button("🔄 Reset Session"):
    st.session_state.trader = None
    st.session_state.runner_thread = None
    st.session_state.show_summary = False
    st.session_state.last_summary = {}
    st.session_state.last_logs = []
    st.session_state.last_timeline = []
    st.rerun()

# Auto-refresh
time.sleep(1)
st.rerun()
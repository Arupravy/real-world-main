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
import plotly.express as px
from email.message import EmailMessage
from io import StringIO

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import plotly.io as pio
import io

from backtesting_engine.real_time_runner import RealTimeTrader
from price_engine.data_sources.websocket_handler import start_price_feed




SESSION_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "session_history.json")

def load_history():
    try:
        if os.path.exists(SESSION_HISTORY_FILE):
            with open(SESSION_HISTORY_FILE, "r") as f:
                history = json.load(f)
                # Validate loaded data is a list
                return history if isinstance(history, list) else []
    except Exception as e:
        st.error(f"Error loading history: {e}")
    return []

def save_history(history):
    """Save session history to JSON file with better validation"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(SESSION_HISTORY_FILE), exist_ok=True)
        
        # Validate history data
        if not isinstance(history, list):
            st.error("History must be a list")
            return False
            
        # Convert pandas Timestamps to strings if present
        for item in history:
            if 'Timestamp' in item and pd.notna(item['Timestamp']):
                item['Timestamp'] = str(item['Timestamp'])
        
        with open(SESSION_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2, default=str)  # Handles non-serializable objects
        return True
    except Exception as e:
        st.error(f"Failed to save history: {str(e)}")
        return False

# In save_completed_session(), add validation:
def save_completed_session(trader, symbols, initial_capital, runtime):
    try:
        summary = trader.get_portfolio_summary()
        st.write("Debug - Trader Summary:", summary)  # Debug output
        
        session_info = {
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Symbols": ', '.join(symbols),
            "Initial Capital": float(summary["initial_capital"]),
            "Final Portfolio Value": float(summary["final_portfolio_value"]),
            "PnL": float(summary["final_pnl"]),
            "Cash Balance": float(summary["cash_balance"]),
            "Unrealized PnL": float(summary["unrealized_pnl"]),
            "Duration (s)": int(runtime)
        }
        
        st.write("Debug - Session Info:", session_info)  # Debug output
        
        st.session_state.completed_runs.append(session_info)
        if save_history(st.session_state.completed_runs):
            st.success("Session saved successfully!")
        else:
            st.error("Failed to save session")
    except Exception as e:
        st.error(f"Error in save_completed_session: {str(e)}")


# --- Fixed Email utility function ---

def send_email_with_chart(summary, logs, fig, recipient="divyanshuydv0002@gmail.com"):
    sender_email = "alert.realworld@gmail.com"
    subject = "📈 Final Trading Profit and Loss Report"

    # Load images
    header_img_path = "Images/Real World Header.jpeg"
    footer_img_path = "Images/Real World Footer.jpeg"

    # HTML body with template and images
    html_body = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f9f9f9;
                padding: 20px;
                color: #333;
            }}
            .summary {{
                background-color: #ffffff;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .summary h2 {{
                color: #0066cc;
                margin-bottom: 15px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            td {{
                padding: 8px;
                border-bottom: 1px solid #eee;
            }}
            .logs {{
                background-color: #ffffff;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            ul {{
                padding-left: 20px;
            }}
        </style>
    </head>
    <body>
        <img src="cid:header_img" alt="Header Image" width="100%" />
        <div class="summary">
            <h2>📊 Final Trading Summary</h2>
            <table>
                <tr><td><strong>Final Portfolio Value:</strong></td><td>${summary['final_portfolio_value']:.2f}</td></tr>
                <tr><td><strong>Final Profit/Loss:</strong></td><td>${summary['final_pnl']:.2f}</td></tr>
                <tr><td><strong>Cash Balance:</strong></td><td>${summary['cash_balance']:.2f}</td></tr>
                <!-- Add check for 'unrealized_pn' key -->
                <tr><td><strong>Unrealized Profit/Loss:</strong></td><td>${summary.get('unrealized_pnl', 0.00):.2f}</td></tr>
            </table>
        </div>

        <div class="logs">
            <h2>🧾 Recent Trade Logs</h2>
            <ul>
                {''.join(f"<li>{log}</li>" for log in logs[-10:])}
            </ul>
        </div>

        <p>📎 Attached below is your Profit and Loss chart.</p>
        <img src="cid:footer_img" alt="Footer Image" width="100%" />
    </body>
    </html>
    """

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    try:
        # Convert plotly figure to image
        img_bytes = pio.to_image(fig, format='jpg', engine='kaleido')
        image = MIMEImage(img_bytes, name="pnl_chart.jpg")
        msg.attach(image)

        # Attach header image
        with open(header_img_path, 'rb') as f:
            header_img = MIMEImage(f.read())
            header_img.add_header('Content-ID', '<header_img>')
            msg.attach(header_img)

        # Attach footer image
        with open(footer_img_path, 'rb') as f:
            footer_img = MIMEImage(f.read())
            footer_img.add_header('Content-ID', '<footer_img>')
            msg.attach(footer_img)

        # Send via Mailjet
        with smtplib.SMTP("in-v3.mailjet.com", 587, timeout=10) as server:
            server.starttls()
            server.login("CENSORED", "CENSORED")
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
st.sidebar.title("⚙️ Trading Configuration")

initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=10000)

# Change this line to allow unrestricted runtime input
runtime = st.sidebar.number_input("Runtime (seconds)", min_value=1, value=300, step=10)

symbols = st.sidebar.multiselect(
    "Symbols to Track",
    ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
    default=["BTCUSDT", "ETHUSDT"]
)

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
        save_completed_session(
            st.session_state.trader, 
            symbols, 
            initial_capital, 
            runtime
        )
        
        st.session_state.trader = None
        st.session_state.show_summary = True

# Handle summary display after session ends
if st.session_state.trader is None:
    if st.session_state.get("show_summary"):
        st.success("🎉 Trading session completed!")

        st.title("♞  The Real World Trading Engine")
        st.subheader("📋 Final Summary")
        final_pnl = st.session_state.last_summary.get("final_pnl", 0)
        final_value = st.session_state.last_summary.get("final_portfolio_value", 0)

        st.metric("💰 Final Portfolio Value", f"${final_value:,.2f}")
        st.metric("📊 Final Profit/Loss", f"${final_pnl:,.2f}")

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
        # st.write("DEBUG - completed_runs:", st.session_state.get("completed_runs", "Not Found"))

        if st.session_state.completed_runs:
            st.subheader("📊 Session History")
            history_df = pd.DataFrame(st.session_state.completed_runs)
            st.dataframe(history_df)

            st.subheader("📈 Profit & Loss Distribution")
            pnl_values = [run['PnL'] for run in st.session_state.completed_runs]
            fig = px.histogram(pnl_values, nbins=20, title="Profit & Loss Distribution Across All Sessions")
            st.plotly_chart(fig)

        st.stop()
    else:
        st.title("♞ The Real World Trading Engine")
        st.info("Configure and start The Real World Trading Engine from the sidebar.")
        st.image("Images/Real World 4k.jpeg", use_container_width=True)
        st.title("Money making is A SKILL, we will teach you how to MASTER IT.")
        st.stop()

# Active session continues here
trader = st.session_state.trader

elapsed = int(time.time() - trader.start_time)
remaining = max(runtime - elapsed, 0)
st.sidebar.metric("⏳ Time Remaining", f"{remaining} sec")

# Auto-close if trader ends silently
# Auto-close if trader ends silently
if st.session_state.trader and not st.session_state.trader.is_active:
    trader = st.session_state.trader

    # Capture session data
    st.session_state.last_summary = trader.get_portfolio_summary()
    st.session_state.last_logs = trader.get_logs()
    st.session_state.last_timeline = trader.get_pnl_data()

    # Save session
    save_completed_session(
        trader,
        symbols,
        initial_capital,
        runtime
    )

    # Generate chart only if timeline exists
    pnl_df = pd.DataFrame(st.session_state.last_timeline)
    fig = None
    if not pnl_df.empty and {"timestamp", "portfolio_value"}.issubset(pnl_df.columns):
        pnl_df["timestamp"] = pd.to_datetime(pnl_df["timestamp"])
        fig = go.Figure()
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

    # Send email only if not already sent
    if not st.session_state.get("email_sent", False):
        send_email_with_chart(
            st.session_state.last_summary,
            st.session_state.last_logs,
            fig
        )
        st.session_state.email_sent = True

    # Clean up and trigger summary display
    st.session_state.trader = None
    st.session_state.show_summary = True
    st.rerun()


if time.time() - getattr(trader, 'last_save_time', 0) > 300:  # Auto-save every 5 minutes
    save_history(st.session_state.completed_runs)
    trader.last_save_time = time.time()

# --- Main UI ---
st.title("♞ The Real World Trading Engine")
st.subheader("📊 Real-Time Trading Dashboard")


# Portfolio Summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💼 Initial Capital", f"${trader.initial_capital:,.2f}")
with col2:
    st.metric("💵 Cash Balance", f"${trader.cash_balance:,.2f}")
with col3:
    unrealized = trader.calculate_unrealized_pnl()
    st.metric("📈 Unrealized P&L", f"${unrealized:,.2f}")
with col4:
    portfolio_value = trader.cash_balance + unrealized
    st.metric("📊 Current Portfolio Value", f"${portfolio_value:,.2f}")

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
#C:\real-world-main\src\streamlit_dashboard.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import json
from datetime import datetime, date
import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import threading
import time
import smtplib
import numpy as np
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
from backtesting_engine.real_time_runner_sentiment import RealTimeSentimentTrader
from price_engine.data_sources.websocket_handler import start_price_feed

from backtesting_engine.backtest_runner import run_backtest_for_ui


from sentiment.news_fetcher import fetch_cryptopanic_news
from sentiment.improved_sentiment_module import SentimentTracker
from sentiment.improved_sentiment_module import (
    EnhancedSentimentAnalyzer,
    SentimentTracker as SmoothedSentimentTracker
)


from price_engine import aggregator
from price_engine.aggregator import aggregate_price

from price_engine.price_history import PriceHistory
from price_engine.data_sources.yahoo_finance import YahooFinanceAPI


import streamlit as st

if "page" not in st.session_state:
    st.session_state.page = "main"



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
            "Duration (s)": int(runtime),
            "Mode": (
                "Sentiment-Based"
                if isinstance(trader, RealTimeSentimentTrader)
                else "Trend-Breakout"
            )
        }
        
        st.write("Debug - Session Info:", session_info)  # Debug output
        
        st.session_state.completed_runs.append(session_info)
        if save_history(st.session_state.completed_runs):
            st.success("Session saved successfully!")
        else:
            st.error("Failed to save session")
    except Exception as e:
        st.error(f"Error in save_completed_session: {str(e)}")

# portfolio timeline function
def display_portfolio_timeline(
    completed_runs,
    title="Portfolio Performance Timeline",
    height=500,
    line_color='#636EFA',
    marker_size_multiplier=2,
    colorscale='RdYlGn',
    padding=None,
    show_legend=True,
    show_dataframe=False
):
    """
    Displays an interactive portfolio performance timeline with trade markers
    
    Args:
        completed_runs (list): List of trading session dictionaries
        title (str): Chart title
        height (int): Chart height in pixels
        line_color (str): Color for the portfolio value line
        marker_size_multiplier (float): Adjusts trade marker sizes
        colorscale (str): Plotly colorscale name for PnL coloring
        padding (dict): Custom padding dict (default: {l:40, r:40, t:80, b:40})
        show_legend (bool): Whether to show the legend
        show_dataframe (bool): Whether to show the raw data table
    """
    # Default padding
    if padding is None:
        padding = dict(l=40, r=40, t=80, b=40)
    
    # Create DataFrame and calculate metrics
    df = pd.DataFrame(completed_runs).copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Return_Pct'] = (df['PnL'] / df['Initial Capital']) * 100
    
    # Categorize trade sizes
    bins = [0, 9999, 99999, float('inf')]
    labels = ['Small', 'Medium', 'Large']
    df['Trade_Size'] = pd.cut(df['Initial Capital'], bins=bins, labels=labels)
    
    # Create figure
    fig = go.Figure()
    
    # Add portfolio value line
    fig.add_trace(go.Scatter(
        x=df['Timestamp'],
        y=df['Final Portfolio Value'],
        name='Portfolio Value',
        line=dict(color=line_color, width=2),
        mode='lines',
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>Value: $%{y:,.2f}'
    ))
    
    # Add trade markers
    fig.add_trace(go.Scatter(
        x=df['Timestamp'],
        y=df['Final Portfolio Value'],
        name='Trades',
        mode='markers',
        marker=dict(
            size=np.log(df['Initial Capital']) * marker_size_multiplier,
            color=df['PnL'],
            colorscale=colorscale,
            colorbar=dict(
                title='PnL ($)',
                x=1.05,
                xpad=10
            ),
            line=dict(width=1, color='DarkSlateGrey'),
            sizemode='diameter'
        ),
        customdata=np.stack((
            df['Symbols'],
            df['Initial Capital'],
            df['PnL'],
            df['Return_Pct'],
            df['Duration (s)'],
            df['Trade_Size']
        ), axis=-1),
        hovertemplate=(
            '<b>%{x|%Y-%m-%d %H:%M}</b><br>'
            'Symbols: <b>%{customdata[0]}</b><br>'
            'Size: <b>%{customdata[5]}</b><br>'
            'Initial: $%{customdata[1]:,.2f}<br>'
            'PnL: $%{customdata[2]:,.2f}<br>'
            'Return: %{customdata[3]:.2f}%<br>'
            'Duration: %{customdata[4]}s'
        )
    ))
    
    # Update layout with perfect padding
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Time",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        template='plotly_dark',
        height=height,
        showlegend=show_legend,
        margin=padding,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Display components
    st.plotly_chart(fig, use_container_width=True)
    
    if show_dataframe:
        with st.expander("View Raw Data"):
            st.dataframe(df.sort_values('Timestamp', ascending=False))


#symbol performance chart on summary page

def display_symbol_performance(history_df):
    """
    Displays comprehensive symbol performance analysis
    
    Args:
        history_df (pd.DataFrame): DataFrame containing trading history with columns:
            ['Timestamp', 'Symbols', 'Initial Capital', 'Final Portfolio Value', 
             'PnL', 'Cash Balance', 'Unrealized PnL', 'Duration (s)']
    """
    # Create analysis DataFrame with calculated metrics
    analysis_df = history_df.copy()
    analysis_df['Return_Pct'] = (analysis_df['PnL'] / analysis_df['Initial Capital']) * 100
    analysis_df['Trade_Result'] = np.where(analysis_df['PnL'] >= 0, 'Profit', 'Loss')
    
    # Calculate symbol statistics
    symbol_stats = analysis_df.groupby('Symbols').agg({
        'Initial Capital': ['count', 'mean'],
        'PnL': ['sum', 'mean', 'median'],
        'Return_Pct': 'mean',
        'Duration (s)': 'median'
    }).reset_index()
    
    # Flatten multi-index columns
    symbol_stats.columns = [
        'Symbols', 'Trade_Count', 'Avg_Capital',
        'Total_PnL', 'Avg_PnL', 'Median_PnL', 
        'Avg_Return_Pct', 'Median_Duration'
    ]
    symbol_stats['Win_Rate'] = analysis_df.groupby('Symbols')['PnL'].apply(
        lambda x: (x >= 0).mean() * 100
    ).values
    
    # Create tabbed interface
    tab1, tab2 = st.tabs(["📈 Performance Metrics", "💰 Capital Efficiency"])
    
    with tab1:
        # --- PnL Distribution ---
        st.markdown("### PnL Distribution by Symbol")
        fig1 = px.box(
            analysis_df,
            x='Symbols',
            y='PnL',
            color='Symbols',
            points=False,
            height=400
        )
        fig1.update_layout(
            showlegend=False,
            yaxis_title="Profit/Loss ($)",
            xaxis_title="",
            margin=dict(t=30, b=30, l=20, r=20)
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # --- Key Metrics ---
        st.markdown("### Key Performance Metrics")
        st.dataframe(
            symbol_stats[['Symbols', 'Trade_Count', 'Avg_PnL', 'Win_Rate', 'Avg_Return_Pct']]
            .sort_values('Avg_PnL', ascending=False)
            .style.format({
                'Avg_PnL': '${:,.2f}',
                'Win_Rate': '{:.1f}%',
                'Avg_Return_Pct': '{:.2f}%'
            }),
            height=300,
            use_container_width=True
        )
    
    with tab2:
        st.markdown("### Performance vs Capital Allocation")
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig3 = px.density_heatmap(
                analysis_df,
                x='Symbols',
                y='Return_Pct',
                z='Initial Capital',
                histfunc="avg",
                nbinsx=min(10, len(analysis_df['Symbols'].unique())),
                nbinsy=10,
                color_continuous_scale='Viridis',
                height=400
            )
            fig3.update_layout(
                yaxis_title="Return %",
                xaxis_title="",
                margin=dict(t=30)
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.markdown("### Win Rate by Symbol")
            fig2 = px.bar(
                symbol_stats.sort_values('Win_Rate', ascending=False),
                x='Symbols',
                y='Win_Rate',
                color='Symbols',
                text_auto='.1f',
                height=400
            )
            fig2.update_layout(
                showlegend=False,
                yaxis_title="Win Rate %",
                xaxis_title="",
                margin=dict(t=30))
            st.plotly_chart(fig2, use_container_width=True)

    # Add spacing at bottom
    st.markdown("<br>", unsafe_allow_html=True)


# --- Fixed Email utility function ---

def send_email_with_chart(summary, logs, fig, recipient="arupravy3@gmail.com"):
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

# Initialize all critical session states
if 'trader' not in st.session_state:
    st.session_state.trader = None
    st.session_state.runner_thread = None
    st.session_state.last_summary = {}
    st.session_state.last_logs = []
    st.session_state.last_timeline = []
    st.session_state.show_summary = False
    st.session_state.completed_runs = load_history()
    st.session_state.email_sent = False
    st.session_state.session_saved = False
    st.session_state.is_running = False

page = st.sidebar.radio("🧭 Select Mode", ["Real-Time Trading", "Backtesting", "Price Engine"])

if page == "Real-Time Trading":
    st.session_state.page = "trading"

elif page == "Backtesting":
    st.session_state.page = "backtest"

elif page == "Price Engine":
    st.session_state.page = "price_engine"


# Sidebar config
# === Real-Time Trading UI ===
if st.session_state.page == "trading":
    st.sidebar.title("⚙️ Trading Configuration")
    
    # — new sentiment toggle —
    sentiment_on = st.sidebar.checkbox(
        "Enable Sentiment-Based Filtering",
        value=False,
        key="sentiment_on",
        help="If on, only trade when EMA sentiment crosses your thresholds"
    )

    symbols = st.sidebar.multiselect(
        "Symbols to Track",
        ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", 
        "MATICUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT", "LINKUSDT", "AVAXUSDT",
        "MKRUSDT", "FTMUSDT", "GALAUSDT", "OPUSDT", "ETHBTC"],
        default=["BTCUSDT", "ETHUSDT"],
        key="symbols"
    )

    initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=10000)
    runtime = st.sidebar.number_input("Runtime (seconds)", min_value=1, value=300, step=10)

    # Start button
    if st.sidebar.button("▶️ Start Trading") and not st.session_state.is_running:
        with st.spinner("🚀 Starting trading session..."):
            TraderClass = (
                RealTimeSentimentTrader if st.session_state.sentiment_on
                else RealTimeTrader
            )
            st.session_state.trader = TraderClass(
                capital=initial_capital,
                runtime=runtime,
                long_thresh=0.20,
                short_thresh=-0.10
            )

            if st.session_state.runner_thread is None or not st.session_state.runner_thread.is_alive():
                st.session_state.runner_thread = threading.Thread(
                    target=start_price_feed,
                    args=(symbols, st.session_state.trader.on_price_update),
                    daemon=True
                )
                st.session_state.runner_thread.start()

            st.session_state.is_running = True
            st.session_state.show_summary = False
            st.session_state.trader.is_active = True
            st.session_state.trader.start_time = time.time()
            st.success("Trading session ACTIVE - Live data streaming")
            st.rerun()

    # Stop button
    if st.sidebar.button("⏹ Stop Trading"):
        if st.session_state.trader and st.session_state.is_running:
            with st.spinner("🛑 Stopping session..."):
                trader = st.session_state.trader
                trader.stop()

                st.session_state.last_summary = trader.get_portfolio_summary()
                st.session_state.last_logs = trader.get_logs()
                st.session_state.last_timeline = trader.get_pnl_data()

                save_completed_session(trader, symbols, initial_capital, runtime)

                st.session_state.is_running = False
                st.session_state.show_summary = True
                st.rerun()

    # Check for expired sessions
    if (st.session_state.is_running and 
        st.session_state.trader and 
        not st.session_state.trader.is_active):
        
        with st.spinner("📊 Preparing session summary..."):
            trader = st.session_state.trader
            st.session_state.last_summary = trader.get_portfolio_summary()
            st.session_state.last_logs = trader.get_logs()
            st.session_state.last_timeline = trader.get_pnl_data()
            
            save_completed_session(trader, symbols, initial_capital, runtime)
            st.session_state.is_running = False
            st.session_state.show_summary = True
            st.rerun()

    # Handle summary display after session ends
    if st.session_state.show_summary:
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
                st.plotly_chart(fig, use_container_width=True)

                peak_value = pnl_df["portfolio_value"].max()
                peak_time = pnl_df.loc[pnl_df["portfolio_value"].idxmax(), "timestamp"]
                st.markdown(
                    f"""<p style='font-size:18px; color:#f1f1f1; margin-top:1.5rem;'>
                        <strong>🔝 Peak Portfolio Value:</strong> 
                        <span style='color:#27ae60;'>${peak_value:,.2f}</span> 
                        <span style='color:#aaaaaa;'>on {peak_time.strftime('%Y-%m-%d %H:%M:%S')}</span>
                    </p>""",
                    unsafe_allow_html=True
                )

            if st.button("📧 Send Final Email Summary"):
                if send_email_with_chart(st.session_state.last_summary, 
                                    st.session_state.last_logs, 
                                    fig):
                    st.success("✅ Email sent successfully!")
                else:
                    st.error("❌ Failed to send email")
            
            if st.sidebar.button("🔄 Reset Session"):
                st.session_state.trader = None
                st.session_state.runner_thread = None
                st.session_state.show_summary = False
                st.session_state.last_summary = {}
                st.session_state.last_logs = []
                st.session_state.last_timeline = []
                st.rerun()
            
            # Display Session History Table
            # st.write("DEBUG - completed_runs:", st.session_state.get("completed_runs", "Not Found"))
            
            #Sentiment
            st.subheader("📰 Sentiment Analysis of the latest news")

            # instantiate once per session
            analyzer = EnhancedSentimentAnalyzer(use_transformer=False)
            tracker  = SmoothedSentimentTracker(max_len=100, ema_alpha=0.3)

            def extract_crypto_symbols(usdt_pairs):
                return list(set([s.replace("USDT", "") for s in usdt_pairs if s.endswith("USDT")]))

            try:
                selected_symbols = st.session_state.get("symbols", [])
                crypto_keywords = extract_crypto_symbols(selected_symbols)
                currencies = ",".join(crypto_keywords) if crypto_keywords else "BTC,ETH"

                news_items = fetch_cryptopanic_news( 
                    filter="trending",
                    limit=5,
                    currencies=currencies
                )

                # score & update history
                for post in news_items:
                    title = post.get("title", "")
                    # cryptopanic tags are dicts, so extract slug or name:
                    tags  = [t.get("slug", "") for t in post.get("tags", [])]
                    score = analyzer.compute_sentiment(title, tags)
                    tracker.update(score, title)

                st.metric("📊 Raw Avg Sentiment", round(tracker.get_average_sentiment(), 3))
                st.metric("📉 EMA Sentiment",     round(tracker.get_ema_sentiment(),     3))

                st.markdown("### 🧾 Latest Headlines with Sentiment")
                for post, item in zip(news_items, tracker.get_history()[-len(news_items):]):
                    emoji  = "🟢" if item["score"] > 0.3 else "🟡" if item["score"] > 0 else "🔴"
                    source = post.get("source", {}).get("title", "Unknown")
                    st.write(f"**{emoji} {item['headline']}**")
                    st.caption(
                        f"🕒 {item['timestamp']} — Source: {source} — Sentiment: {item['score']:.4f}"
                    )
                    st.markdown("---")

                if st.session_state.completed_runs:
                    st.subheader("📊 Session History")
                    history_df = pd.DataFrame(st.session_state.completed_runs)
                    st.dataframe(history_df)

                    st.subheader("📈 Portfolio Performance Timeline")
                    display_portfolio_timeline(
                        st.session_state.completed_runs,
                        title="Trading Performance",
                        height=600,
                        line_color="#00CC96",
                        colorscale="Viridis",
                        padding=dict(l=50, r=50, t=100, b=50),
                        show_dataframe=True
                    )
                    display_symbol_performance(history_df)

                st.stop()

            except Exception as e:
                st.warning(f"Could not load CryptoPanic news: {e}")

    # Active trading UI - ONLY show if trader exists

    if st.session_state.is_running and st.session_state.trader:
        trader = st.session_state.trader

        st.sidebar.checkbox("Show Debug Info", key="show_debug")

        # 2) Only debug info lives inside this guard
        if st.session_state.show_debug:
            st.sidebar.subheader("🧑‍💻 Trader State Validation")
            val = trader.validate_state()
            st.sidebar.json({
                "Status":        "🟢 ACTIVE" if val["is_active"] else "🔴 INACTIVE",
                "Runtime":       f"{val['last_update']:.1f}s",
                "Open Positions": val["positions"],
                "Thread Status": "Alive" if st.session_state.runner_thread.is_alive() else "Dead"
            })
            if st.sidebar.button("Force Refresh State", key="force_refresh"):
                st.experimental_rerun()

        # 3) If you have sentiment trader, show its live EMA
        if isinstance(trader, RealTimeSentimentTrader):
            st.sidebar.metric(
                "📉 Live EMA Sentiment",
                f"{trader.get_sentiment_ema():.3f}"
            )

        # 4) These two always display exactly once
        elapsed   = int(time.time() - trader.start_time)
        remaining = max(trader.runtime - elapsed, 0)
        st.sidebar.metric("⏳ Time Remaining", f"{remaining} sec")

        # 5) **Session timeout → show summary**
        if remaining <= 0:
            st.warning("Session completed - preparing summary…")
            time.sleep(1)
            st.rerun()              # rerun back into your summary branch

        # 6) **Auto-save every 5 minutes**
        if time.time() - getattr(trader, "last_save_time", 0) > 300:
            save_history(st.session_state.completed_runs)
            trader.last_save_time = time.time()

        if st.sidebar.button("🔄 Reset Session", key="reset_session"):
            st.session_state.trader        = None
            st.session_state.runner_thread = None
            st.session_state.is_running    = False
            st.session_state.show_summary  = False
            st.session_state.last_summary  = {}
            st.session_state.last_logs     = []
            st.session_state.last_timeline = []
            st.rerun()

        # --- now your main UI continues below ---
        st.title("♞ The Real World Trading Engine")
        st.subheader("📊 Real-Time Trading Dashboard")


        # Portfolio Summary
        col1, col2, col3, col4 = st.columns([2,2,1,2])
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

        # Safely retrieve symbols from session_state

        if st.session_state.sentiment_on \
            and isinstance(trader, RealTimeSentimentTrader) \
            and st.session_state.symbols:

            st.subheader("🔎 Symbol-Level Sentiment Breakdown")
            cols = st.columns(len(st.session_state.symbols))
            analyzer = EnhancedSentimentAnalyzer(use_transformer=False)

            for i, sym in enumerate(st.session_state.symbols):
                tracker_sym = SentimentTracker(max_len=20, ema_alpha=0.3)
                posts = fetch_cryptopanic_news(
                    filter="trending", limit=5, currencies=sym.replace("USDT","")
                ) or []
                for post in posts:
                    title = post["title"]
                    tags  = [t.get("slug","") for t in post.get("tags",[])]
                    tracker_sym.update(analyzer.compute_sentiment(title,tags), title)

                cols[i].metric(
                    sym,
                    f"Avg: {tracker_sym.get_average_sentiment():.3f}",
                    f"EMA: {tracker_sym.get_ema_sentiment():.3f}"
                )

            # Live EMA Sentiment Chart (overall)
            st.subheader("📈 EMA Sentiment Over Time")
            history = trader.tracker.get_history()
            if history:
                df = pd.DataFrame(history)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
                plot_df = pd.DataFrame({
                    "Raw": df["score"],
                    "EMA": df["score"].ewm(alpha=trader.tracker.ema_alpha).mean()
                })
                st.line_chart(plot_df, use_container_width=True)

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

        # Auto-refresh
        time.sleep(1)
        st.rerun()

    else:
        # Initial state when no trader exists
        st.title("♞ The Real World Trading Engine")
        st.info("Configure and start The Real World Trading Engine from the sidebar.")
        st.image("Images/Real World 4k.jpeg")
        st.title("Money making is A SKILL, we will teach you how to MASTER IT.")





# === Backtesting UI ===
elif st.session_state.page == "backtest":
    st.title("🧠 Backtesting Interface")

    # Sidebar config for backtesting
    st.sidebar.title("🧠 Backtesting Configuration")

    bt_initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=100000, step=1000)
    asset_type = st.sidebar.radio("Select Asset Type", ["crypto", "stock"], horizontal=True)
 
    # Full list of all symbols (crypto and stocks)
    all_symbols = [
        # Crypto
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "XRPUSDT", "LTCUSDT", "DOTUSDT", 
        "MATICUSDT", "LINKUSDT", "TRXUSDT", "AVAXUSDT", "SHIBUSDT", "FTMUSDT", "NEARUSDT", "AAVEUSDT", "ALGOUSDT", 
        "GRTUSDT", "FILUSDT", "STXUSDT", "FTTUSDT", "XLMUSDT", "LUNAUSDT", "CROUSDT", "ZRXUSDT", "OCEANUSDT", 
        "SUSHIUSDT", "EGLDUSDT", "KSMUSDT", "MKRUSDT", "MANAUSDT", "SANDUSDT", "AUDIOUSDT", "DODOUSDT", "ZECUSDT", 
        "BANDUSDT", "RUNEUSDT", "CVCUSDT", "LENDUSDT", "STPTUSDT", "BNTUSDT", "SNTUSDT", "CRVUSDT", "YFIUSDT", 
        "1INCHUSDT", "COMPUSDT", "UNFIUSDT", "SPELLUSDT", "MASKUSDT", "KNCUSDT", "FARMUSDT", "MITHUSDT", "FLOKIUSDT", 
    
        # Stocks
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "BRK.B", "JPM", "V", "WMT", "BA", 
        "DIS", "PYPL", "BABA", "UBER", "SNAP", "SHOP", "INTC", "AMD", "NVDA", "CSCO", "ORCL", "SPY", "SP500", 
        "DIA", "IWM", "QQQ", "XOM", "CVX", "GE", "KO", "PEP", "MCD", "JNJ", "PG", "NKE", "HD", "LOW", "MS", 
        "GS", "C", "BMO", "TD", "RBC", "WFC", "AXP", "MA", "SQ", "GOOG", "LULU", "VZ", "T", "GS", "BA", "COST", 
        "AMAT", "LRCX", "CAT", "ADM", "MMM", "RTX", "HON", "TXN", "AVGO", "PFE", "MRK", "MDT", "ABBV", "AMGN", 
        "BIIB", "ISRG", "GILD", "SYK", "REGN", "VRTX", "VEEV", "IQV", "IDXX", "RMD", "TMO", "ABT", "ZBH", 
        "COO", "SYF", "HUM", "UNH", "CVS", "ANTM", "MCK", "WBA", "TGT", "FISV", "CME", "ICE", "NDAQ", "MSCI", 
        "SPGI", "LMT", "RTX", "HII", "NOC", "GD", "HUM", "CNC", "KHC", "K", "PG", "NKE", "TGT"
    ]

    # Show popular symbols first
    popular_symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
        "META", "NVDA", "NFLX", "BRK.B", "JPM", "V", "WMT"
    ]

    # Dropdown menu for popular symbols
    selected_symbols = st.sidebar.multiselect(
        "Select Symbols to Backtest (Popular)",
        options=popular_symbols,
        default=["BTCUSDT", "ETHUSDT"]
    )

    # Text input for custom symbols
    custom_symbols = st.sidebar.text_input("Or Enter Custom Symbols (comma-separated)", "")

    # Combine custom symbols with popular symbols
    if custom_symbols:
        custom_symbol_list = [symbol.strip().upper() for symbol in custom_symbols.split(",")]
        selected_symbols = list(set(selected_symbols) | set(custom_symbol_list))

    start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date(2024, 1, 1))

    strategy = st.selectbox("Select Strategy", options=["bollinger", "mean_reversion", "trendpullback"])

    default_allocations = [round(100 / len(selected_symbols), 2) for _ in selected_symbols] if selected_symbols else []

    allocations_input = st.sidebar.text_input(
        "Allocations (%) for each symbol (comma-separated)",
        value=",".join(map(str, default_allocations)) if default_allocations else ""
    )

    run_btn = st.sidebar.button("🚀 Run Backtest")

    if run_btn:
        if not selected_symbols:
            st.warning("⚠️ Please select at least one symbol.")
        else:
            try:
                # Parse allocations and validate
                allocations = list(map(float, allocations_input.split(",")))

                if len(allocations) != len(selected_symbols):
                    st.error("❌ Number of allocations must match number of selected symbols.")
                elif round(sum(allocations), 2) != 100.0:
                    st.error("❌ Allocations must sum to 100%.")
                else:
                    with st.spinner("Running backtest..."):
                        results = run_backtest_for_ui(
                            symbols=selected_symbols,
                            allocations=allocations,
                            start=start_date.strftime("%Y-%m-%d"),
                            end=end_date.strftime("%Y-%m-%d"),
                            strategy=strategy,
                            initial_capital=bt_initial_capital,
                            asset_type=asset_type  # ✅ add this
                        )


                        st.success("✅ Backtest Complete!")
                        st.metric("Initial Capital", f"${results['initial_capital']:,}")
                        st.metric("Final Net Worth", f"${results['final_net_worth']:,.2f}")
                        st.metric("Total Return", f"{results['total_return']:.2f}%")
                        st.metric("Total Trades", results["total_trades"])

                        st.subheader("📊 Per-Symbol Performance")
                        st.dataframe(results["per_symbol_logs"])

                        st.subheader("🧾 Trade Log")
                        st.dataframe(results["trade_log_df"])

                        if results["pnl_chart_data"] is not None:
                            st.line_chart(results["pnl_chart_data"])

                        # === 📊 Interactive Bar: Per Symbol Final Net Worth ===
                        st.subheader("📊 Per-Symbol Final Net Worth")
                        per_symbol_df = pd.DataFrame(results["per_symbol_logs"])
                        per_symbol_fig = go.Figure()

                        for symbol in selected_symbols:
                            symbol_data = per_symbol_df[per_symbol_df["symbol"] == symbol]
                            per_symbol_fig.add_trace(go.Bar(
                                x=[symbol],
                                y=symbol_data["final_net_worth"],
                                name=symbol
                            ))

                        per_symbol_fig.update_layout(
                            title="Final Net Worth per Symbol",
                            xaxis_title="Symbols",
                            yaxis_title="Final Net Worth",
                            barmode="group"
                        )
                        st.plotly_chart(per_symbol_fig, use_container_width=True)

                        # === 📈 PnL Over Time (Cleaned Layout) ===
                        if results.get("pnl_chart_data") is not None:
                            st.subheader("📉 PnL Over Time")
                            pnl_fig = go.Figure()
                            pnl_fig.add_trace(go.Scatter(
                                x=results["pnl_chart_data"].index,
                                y=results["pnl_chart_data"],
                                mode="lines+markers",
                                name="Net Worth"
                            ))
                            pnl_fig.update_layout(
                                title="Portfolio Net Worth Over Time",
                                xaxis_title="Date",
                                yaxis_title="Net Worth ($)"
                            )
                            st.plotly_chart(pnl_fig, use_container_width=True)
                            

                        # === 🔥 Trade Volume Heatmap ===
                        st.subheader("🔥 Trade Volume Heatmap")
                        if not results["trade_log_df"].empty:
                            trade_counts = results["trade_log_df"].groupby(["symbol", "action"]).size().unstack(fill_value=0)
                            heatmap_fig = go.Figure(data=go.Heatmap(
                                z=trade_counts.values,
                                x=trade_counts.columns.tolist(),
                                y=trade_counts.index.tolist(),
                                colorscale="Viridis"
                            ))
                            heatmap_fig.update_layout(
                                xaxis_title="Action",
                                yaxis_title="Symbol",
                                title="Trade Volume Heatmap (Buy/Sell Frequency)"
                            )
                            st.plotly_chart(heatmap_fig, use_container_width=True)
                        else:
                            st.info("No trades were executed, so heatmap is not available.")

            except Exception as e:
                st.error(f"❌ Error during backtest: {e}")

# 🧠 Price Engine Dashboard
elif st.session_state.page == "price_engine":

    from price_engine.indicators.bollinger_bands import BollingerBands
    from price_engine.indicators.mean_reversion import MeanReversion
    from price_engine.indicators.enhanced_mean_reversion import EnhancedMeanReversion

    from price_engine.data_sources import (
        binance_api,
        yahoo_finance,
        coingecko_api,
        kraken_api,
        coinbase_api
    )



    st.title("⚙️ Price Engine Dashboard")

    # === UI Elements ===

    # Popular stock symbols
    stock_symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "INTC", "AMD",
        "SPY", "QQQ", "BABA", "V", "PYPL", "DIS", "BA", "IBM", "GS", "WMT", "XOM"
    ]

    # Popular crypto symbols
    crypto_symbols = [
        "BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", 
        "MATICUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT", "LINKUSDT", "AVAXUSDT",
        "MKRUSDT", "FTMUSDT", "GALAUSDT", "OPUSDT", "ETHBTC"
    ]

    # Streamlit UI components
    symbol = st.selectbox("Select Symbol", stock_symbols + crypto_symbols)
    asset_type = st.radio("💼 Select Asset Type", ["stock", "crypto"], horizontal=True)

    mode = st.radio("⚙️ Select Mode", [
        "Live (Aggregated + Indicators)",
        "API Live (Polling)",
        "WebSocket Live (Binance)",
        "Historical (with optional plot)",
        "Live Plot"  # ✅ Added here
    ])

    # Additional controls for Historical Mode
    if mode == "Historical (with optional plot)":
        from_date = st.date_input("📅 From Date", value=date(2024, 1, 1))
        to_date = st.date_input("📅 To Date", value=date.today())  # ✅ Automatically today
        show_plot = st.checkbox("📊 Show Plot", value=True)


    # Additional controls for Live Plot mode
    if mode == "Live Plot":
        selected_symbols = st.multiselect("🧵 Select Symbols", stock_symbols + crypto_symbols, default=["BTCUSDT"])
        duration_seconds = st.slider("⏱️ Duration (seconds)", min_value=30, max_value=300, value=120, step=10)

    st.markdown("---")


    # === Logic for Each Mode ===

    def run_aggregated_live(symbol, asset_type):
        st.subheader(f"🔁 Aggregated Live Price: {symbol}")

        # Get aggregated price and all source prices
        price, source_prices = aggregate_price(symbol, asset_type)


        if price:
            # Display individual source prices
            st.text(f"Live Prices for {symbol} from all sources:")
            for source, p in source_prices.items():
                st.text(f"{source}: {p}")
            st.success(f"Weighted Average Price: ${round(price, 4)}")
        else:
            st.error("Failed to aggregate price.")

        st.subheader("📈 Indicators")
        dummy_prices = [{"price": price}] * 30 if price else [{"price": 0}] * 30

        # Bollinger Bands
        bb_indicator = BollingerBands()
        bb = bb_indicator.calculate(dummy_prices)

        # Mean Reversion
        mr_indicator = MeanReversion(window=20, threshold=2.0)
        mr = mr_indicator.calculate(dummy_prices)

        mean_reversion_status = (
            "Overbought" if mr["overbought"] else
            "Oversold" if mr["oversold"] else
            "Neutral"
        )

        # Enhanced Mean Reversion
        emr_indicator = EnhancedMeanReversion(window=20)
        emr_action = emr_indicator.decide(dummy_prices, current_position="none", asset_name=symbol)

        emr_status = {
            "buy": "Buy signal",
            "sell": "Sell signal",
            "short": "Short signal",
            "cover": "Cover short signal"
        }.get(emr_action, "No action")

        # Display
        st.write("**Bollinger Bands:**", bb)
        st.write("**Mean Reversion Status:**", mean_reversion_status)
        st.write("**Enhanced Mean Reversion Action:**", emr_status)

    from price_engine.data_sources.yahoo_finance import YahooFinanceAPI
    yahoo_finance = YahooFinanceAPI()  # ✅ Create an instance
    from price_engine.data_sources.binance_api import BinanceAPI
    from price_engine.data_sources.coingecko_api import CoinGeckoAPI
    from price_engine.data_sources.kraken_api import KrakenAPI
    from price_engine.data_sources.coinbase_api import CoinbaseAPI

    binance_api = BinanceAPI()
    coingecko_api = CoinGeckoAPI()
    kraken_api = KrakenAPI()
    coinbase_api = CoinbaseAPI()


    def run_api_live(symbol, asset_type):
        st.subheader(f"🌐 API Live Price Stream: {symbol}")

        # Use only relevant sources
        if asset_type == "crypto":
            source_order = [binance_api, coingecko_api, kraken_api, coinbase_api]
        else:
            source_order = [yahoo_finance]

        latest_box = st.empty()
        log_box = st.empty()  # ✅ Use this for clean overwrite

        price_log = []
        last_price = None

        for _ in range(30):  # ~2.5 minutes
            price = None
            for source in source_order:
                try:
                    fetched_price = source.get_price(symbol)
                    if fetched_price is not None:
                        price = fetched_price
                        break
                except Exception as e:
                    st.error(f"{source.__class__.__name__} failed: {e}")

            timestamp = datetime.datetime.now().strftime('%H:%M:%S')

            if price is not None:
                # Compare with previous price
                if last_price is None:
                    emoji = "🟢"
                elif price > last_price:
                    emoji = "📈"
                elif price < last_price:
                    emoji = "📉"
                else:
                    emoji = "➖"

                last_price = price
                line = f"{emoji} {timestamp} — Price: ${round(price, 4)}"
                price_log.insert(0, line)

                # Latest price in green
                latest_box.success(line)
            else:
                latest_box.warning("Could not fetch price from any source.")

            # Display log cleanly
            log_html = (
                "<h4>🧾 Price Log</h4>"
                "<div style='max-height: 400px; overflow-y: auto; padding-right:10px;'>"
                + "".join(f"<p style='margin:0'>{entry}</p>" for entry in price_log)
                + "</div>"
            )
            log_box.markdown(log_html, unsafe_allow_html=True)

            time.sleep(5)




    def run_ws_live(symbol):
        st.subheader(f"📡 Binance WebSocket Price Stream for {symbol}")

        st.warning("⚠️ WebSocket mode requires a persistent connection and is not supported directly in Streamlit.")

        st.markdown("""
        #### 🛠️ To run this mode, use the command line:
        ```bash
        python src/main.py --mode ws-live --symbol %s --asset-type crypto
        ```
        """ % symbol)

        st.markdown("""
        ✅ This will open a live Binance WebSocket connection and stream real-time prices directly in your terminal.
        """)

    def run_historical(symbol, asset_type, from_date, to_date, show_plot):
        st.subheader(f"🕰️ Historical Data: {symbol}")
        st.markdown(f"Selected range: **{from_date} → {to_date}**")

        df = None

        if asset_type == "crypto":
            # Load from prices.json
            history = PriceHistory("prices.json")
            data = pd.DataFrame(history.get_history())

            data = data[data["symbol"] == symbol]
            if data.empty:
                st.error("No historical data found for this crypto symbol.")
                return

            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data = data[(data["timestamp"].dt.date >= from_date) &
                        (data["timestamp"].dt.date <= to_date)]

            if data.empty:
                st.error("No data found in the selected date range.")
                return

            df = data.rename(columns={"timestamp": "date"}).sort_values("date").reset_index(drop=True)

        elif asset_type == "stock":
            # Use Yahoo live historical fetch
            yahoo = YahooFinanceAPI()
            records = yahoo.get_historical_prices(symbol, str(from_date), str(to_date))
            if not records:
                st.error("No data returned from Yahoo Finance.")
                return
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])

        else:
            st.error("Unsupported asset type.")
            return

        
        # Display full data in selected range
        st.markdown("### 📋 Historical Data")
        st.dataframe(df, use_container_width=True)


        if "price" in df.columns:
            df['bb'] = df['price'].rolling(window=20).mean()
            df['mr'] = df['price'].rolling(window=10).mean()

            if show_plot:
                st.markdown("### 📊 Price with Indicators")
                st.line_chart(df.set_index("date")[["price", "bb", "mr"]])

        st.markdown("### 📈 Summary Stats")
        st.write(df["price"].describe())

    def run_live_plot(selected_symbols, asset_type, duration_seconds):
        st.subheader("📊 Real-Time Live Plot")

        if asset_type == "crypto":
            source_order = [binance_api, coingecko_api, kraken_api, coinbase_api]
        else:
            source_order = [yahoo_finance]

        plot_placeholder = st.empty()
        stop_button = st.button("⏹ Stop Plot")

        # Use session state to manage stopping
        if "stop_plot" not in st.session_state:
            st.session_state.stop_plot = False

        if stop_button:
            st.session_state.stop_plot = True

        data_log = {symbol: [] for symbol in selected_symbols}

        start_time = time.time()

        while not st.session_state.stop_plot and (time.time() - start_time) < duration_seconds:
            now = datetime.datetime.now()

            for symbol in selected_symbols:
                price = None
                for source in source_order:
                    try:
                        fetched_price = source.get_price(symbol)
                        if fetched_price is not None:
                            price = fetched_price
                            break
                    except Exception as e:
                        st.error(f"{source.__class__.__name__} failed for {symbol}: {e}")

                if price is not None:
                    data_log[symbol].append({"time": now, "price": price})

            # Create the plot
            fig = go.Figure()
            for symbol in selected_symbols:
                df = pd.DataFrame(data_log[symbol])
                if not df.empty:
                    fig.add_trace(go.Scatter(x=df["time"], y=df["price"], mode="lines+markers", name=symbol))

            fig.update_layout(title="Live Prices", xaxis_title="Time", yaxis_title="Price")
            plot_placeholder.plotly_chart(
                fig,
                use_container_width=True
            )

            time.sleep(1)  # Adjustable polling rate

        # Reset stop state
        st.session_state.stop_plot = False




    # === Execution Trigger ===
    if st.button("🚀 Run Price Engine"):
        if mode == "Live (Aggregated + Indicators)":
            run_aggregated_live(symbol, asset_type)
        elif mode == "API Live (Polling)":
            run_api_live(symbol, asset_type)
        elif mode == "WebSocket Live (Binance)":
            if asset_type != "crypto":
                st.error("WebSocket mode is only available for crypto assets.")
            else:
                run_ws_live(symbol)
        elif mode == "Historical (with optional plot)":
            run_historical(symbol, asset_type, from_date, to_date, show_plot)
        elif mode == "Live Plot":
            run_live_plot(selected_symbols, asset_type, duration_seconds)





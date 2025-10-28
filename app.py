"""
Financial Sentiment-Driven Stock Prediction System
A complete deep learning web application combining sentiment analysis and LSTM for stock prediction.
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from models.sentiment_analyzer import FinancialSentimentAnalyzer
from models.stock_predictor import LSTMStockPredictor
from utils.news_fetcher import NewsFetcher
from utils.data_preprocessor import StockDataPreprocessor

# Page configuration
st.set_page_config(
    page_title="Stock Prediction System",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    h1 {
        color: #00FFAA;
        text-align: center;
        font-size: 3em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        padding: 20px;
    }
    h2, h3 {
        color: #FFFFFF;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background: linear-gradient(90deg, #00FFAA 0%, #00D4FF 100%);
        color: #000000;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 25px;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,255,170,0.4);
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Title with emoji
st.markdown("<h1>💹 Financial Sentiment-Driven Stock Predictor 📈</h1>", unsafe_allow_html=True)

# Sidebar with project information
with st.sidebar:
    st.markdown("### 📊 About This Project")
    st.markdown("""
    This advanced system combines:
    
    🧠 **Deep Learning LSTM Model**
    - Analyzes historical price patterns
    - Predicts future price movements
    - Uses 60-day lookback window
    
    💬 **Sentiment Analysis**
    - Processes financial news headlines
    - Extracts market sentiment
    - Combines with price data
    
    📈 **Features**
    - Real-time stock data via Yahoo Finance
    - Interactive visualizations
    - Predicted vs Actual price comparison
    - Sentiment trend analysis
    
    ---
    
    ### 🔧 How It Works
    
    1. **Data Collection**: Fetches 1-year historical stock data
    2. **Sentiment Analysis**: Analyzes recent news headlines
    3. **Feature Engineering**: Creates technical indicators
    4. **LSTM Prediction**: Trains model and makes predictions
    5. **Visualization**: Displays results with interactive charts
    
    ---
    
    ### 💡 Tech Stack
    - Python + Streamlit
    - TensorFlow/Keras (LSTM)
    - Scikit-learn
    - Plotly (Interactive Charts)
    - YFinance (Free Stock Data)
    
    ---
    
    **100% Free & Open Source** ✨
    """)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False

# Main content area
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    ticker = st.text_input(
        "🔍 Enter Stock Symbol",
        value="AAPL",
        help="Enter stock ticker (e.g., AAPL, TSLA, GOOGL, MSFT, INFY.NS)"
    )

with col2:
    period = st.selectbox(
        "📅 Time Period",
        options=["1y", "6mo", "3mo", "2y"],
        index=0
    )

with col3:
    forecast_days = st.slider(
        "🔮 Forecast Days",
        min_value=5,
        max_value=30,
        value=10
    )

# Buttons
col_b1, col_b2, col_b3 = st.columns([1, 1, 2])

with col_b1:
    fetch_button = st.button("📥 Fetch Data", use_container_width=True)

with col_b2:
    predict_button = st.button("🚀 Predict Trend", use_container_width=True)

# Cache data loading
@st.cache_data(ttl=3600)
def load_stock_data(ticker_symbol, period_str):
    """Load stock data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period=period_str)
        if len(df) == 0:
            return None
        df = df.reset_index()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def get_sentiment_analyzer():
    """Get cached sentiment analyzer."""
    return FinancialSentimentAnalyzer()

@st.cache_resource
def get_news_fetcher():
    """Get cached news fetcher."""
    return NewsFetcher()

# Fetch Data Button Logic
if fetch_button:
    with st.spinner(f"🔄 Fetching data for {ticker.upper()}..."):
        # Load stock data
        df = load_stock_data(ticker.upper(), period)
        
        if df is not None and len(df) > 60:
            st.session_state.stock_data = df
            st.session_state.ticker = ticker.upper()
            st.session_state.data_loaded = True
            st.success(f"✅ Successfully loaded {len(df)} days of data for {ticker.upper()}!")
            
            # Fetch news
            news_fetcher = get_news_fetcher()
            news_df = news_fetcher.fetch_news(ticker.upper(), days=14)
            st.session_state.news_data = news_df
            
            # Analyze sentiment
            analyzer = get_sentiment_analyzer()
            sentiments = analyzer.analyze_batch(news_df['headline'].tolist())
            news_df['sentiment_score'] = [s['score'] for s in sentiments]
            news_df['sentiment_label'] = [s['label'] for s in sentiments]
            st.session_state.sentiment_data = news_df
            
        else:
            st.error("❌ Unable to fetch data. Please check the ticker symbol and try again.")
            st.session_state.data_loaded = False

# Display data if loaded
if st.session_state.data_loaded:
    st.markdown("---")
    
    # Display stock information
    df = st.session_state.stock_data
    
    # Metrics
    st.markdown("### 📊 Stock Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    price_change = ((current_price - prev_price) / prev_price) * 100
    
    high_52w = df['Close'].max()
    low_52w = df['Close'].min()
    avg_volume = df['Volume'].mean()
    
    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{price_change:+.2f}%"
        )
    
    with col2:
        st.metric("52W High", f"${high_52w:.2f}")
    
    with col3:
        st.metric("52W Low", f"${low_52w:.2f}")
    
    with col4:
        st.metric("Avg Volume", f"{avg_volume/1e6:.2f}M")
    
    with col5:
        sentiment_avg = st.session_state.sentiment_data['sentiment_score'].mean()
        sentiment_emoji = "😊" if sentiment_avg > 0 else "😐" if sentiment_avg == 0 else "😟"
        st.metric("Sentiment", sentiment_emoji, f"{sentiment_avg:.3f}")
    
    # Price chart
    st.markdown("### 📈 Historical Stock Price")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=500,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment Analysis
    st.markdown("### 💬 Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        sentiment_counts = st.session_state.sentiment_data['sentiment_label'].value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker=dict(colors=['#00FF88', '#FFD700', '#FF6B6B']),
            hole=0.4
        )])
        
        fig_pie.update_layout(
            title="Sentiment Distribution",
            template='plotly_dark',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Sentiment over time
        sentiment_df = st.session_state.sentiment_data.copy()
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        daily_sentiment = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['sentiment_score'],
            mode='lines+markers',
            line=dict(color='#00FFAA', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(0,255,170,0.2)'
        ))
        
        fig_line.update_layout(
            title="Sentiment Trend Over Time",
            template='plotly_dark',
            height=400,
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Recent headlines
    with st.expander("📰 Recent News Headlines"):
        for idx, row in st.session_state.sentiment_data.head(10).iterrows():
            sentiment_color = "#00FF88" if row['sentiment_label'] == 'positive' else "#FFD700" if row['sentiment_label'] == 'neutral' else "#FF6B6B"
            st.markdown(
                f"<div style='padding:10px; margin:5px; background:rgba(255,255,255,0.1); border-left:4px solid {sentiment_color}; border-radius:5px;'>"
                f"<b>{row['headline']}</b><br>"
                f"<small>📅 {row['date']} | Sentiment: {row['sentiment_label'].upper()} ({row['sentiment_score']:.3f})</small>"
                f"</div>",
                unsafe_allow_html=True
            )

# Predict Button Logic
if predict_button and st.session_state.data_loaded:
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Predictions")
    
    with st.spinner("🧠 Training LSTM model and generating predictions..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Prepare data
        status_text.text("📊 Preparing data...")
        progress_bar.progress(20)
        
        df = st.session_state.stock_data.copy()
        
        # Feature columns
        feature_columns = ['Close', 'Volume', 'High', 'Low', 'Open']
        
        # Train model
        status_text.text("🏋️ Training LSTM model...")
        progress_bar.progress(40)
        
        predictor = LSTMStockPredictor(lookback=60, units=50)
        
        try:
            history = predictor.train(
                df,
                feature_columns,
                epochs=30,
                batch_size=32,
                validation_split=0.1
            )
            
            progress_bar.progress(70)
            status_text.text("🔮 Generating predictions...")
            
            # Make predictions on historical data
            predictions = predictor.predict(df, feature_columns)
            
            # Predict future
            future_predictions = predictor.predict_future(df, feature_columns, days=forecast_days)
            
            progress_bar.progress(100)
            status_text.text("✅ Predictions complete!")
            
            st.session_state.predictions = predictions
            st.session_state.future_predictions = future_predictions
            st.session_state.predictions_made = True
            
            # Clear progress indicators
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success("✨ Predictions generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("💡 Try using a different stock symbol or time period.")

# Display predictions
if st.session_state.predictions_made:
    st.markdown("---")
    
    df = st.session_state.stock_data.copy()
    predictions = st.session_state.predictions
    future_predictions = st.session_state.future_predictions
    
    # Predicted vs Actual
    st.markdown("### 📊 Predicted vs Actual Prices")
    
    # Align predictions with dates (skip first 60 due to lookback)
    pred_df = df.iloc[60:60+len(predictions)].copy()
    pred_df['Predicted'] = predictions
    
    fig_pred = go.Figure()
    
    # Actual prices
    fig_pred.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Close'],
        mode='lines',
        name='Actual Price',
        line=dict(color='#00FFAA', width=2)
    ))
    
    # Predicted prices
    fig_pred.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Predicted'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='#FF6B6B', width=2, dash='dash')
    ))
    
    fig_pred.update_layout(
        template='plotly_dark',
        height=500,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Future predictions
    st.markdown(f"### 🔮 Future Price Forecast ({forecast_days} Days)")
    
    # Create future dates
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_predictions
    })
    
    # Calculate trend
    trend_change = ((future_predictions[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
    trend_direction = "📈 BULLISH" if trend_change > 0 else "📉 BEARISH"
    trend_color = "#00FF88" if trend_change > 0 else "#FF6B6B"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Price",
            f"${df['Close'].iloc[-1]:.2f}"
        )
    
    with col2:
        st.metric(
            f"Predicted ({forecast_days}d)",
            f"${future_predictions[-1]:.2f}",
            f"{trend_change:+.2f}%"
        )
    
    with col3:
        st.markdown(
            f"<div style='text-align:center; padding:10px; background:{trend_color}30; border-radius:10px; border:2px solid {trend_color};'>"
            f"<h3 style='color:{trend_color}; margin:0;'>{trend_direction}</h3>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    # Future price chart
    fig_future = go.Figure()
    
    # Historical prices (last 30 days)
    historical_last = df.tail(30).copy()
    fig_future.add_trace(go.Scatter(
        x=historical_last['Date'],
        y=historical_last['Close'],
        mode='lines',
        name='Historical',
        line=dict(color='#00FFAA', width=3)
    ))
    
    # Future predictions
    fig_future.add_trace(go.Scatter(
        x=future_df['Date'],
        y=future_df['Predicted_Price'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#FFD700', width=3, dash='dot'),
        marker=dict(size=8)
    ))
    
    fig_future.update_layout(
        template='plotly_dark',
        height=500,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_future, use_container_width=True)
    
    # Prediction confidence
    st.markdown("### 🎯 Prediction Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    actual_values = pred_df['Close'].values
    predicted_values = pred_df['Predicted'].values
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_values, predicted_values)
    
    with col1:
        st.metric("MAE", f"${mae:.2f}")
    
    with col2:
        st.metric("RMSE", f"${rmse:.2f}")
    
    with col3:
        st.metric("R² Score", f"{r2:.4f}")
    
    with col4:
        accuracy = max(0, (1 - (mae / df['Close'].mean())) * 100)
        st.metric("Accuracy", f"{accuracy:.1f}%")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#FFFFFF; padding:20px;'>"
    "<p>💡 <b>Disclaimer:</b> This is an educational project. Not financial advice. Always do your own research before investing.</p>"
    "<p>Built with ❤️ using Python, Streamlit, TensorFlow & Scikit-learn | 100% Free & Open Source</p>"
    "</div>",
    unsafe_allow_html=True
)

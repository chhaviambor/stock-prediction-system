# Financial Sentiment-Driven Stock Prediction System

## Project Overview
A complete deep learning web application that predicts stock price movements by combining financial news sentiment analysis with LSTM time-series forecasting. Built with Python, Streamlit, TensorFlow, and free APIs.

**Status**: ✅ Fully Functional | **Last Updated**: October 28, 2025

## Key Features

### Core Functionality
- ✅ **Stock Data Fetching**: Real-time historical stock data via yfinance (1y, 6mo, 3mo, 2y periods)
- ✅ **Sentiment Analysis**: Custom financial sentiment analyzer using keyword-based NLP
- ✅ **LSTM Predictions**: Deep learning model for 5-30 day price forecasting
- ✅ **Interactive UI**: Beautiful gradient-themed Streamlit interface with animations
- ✅ **Real-time Visualizations**: Plotly candlestick charts, sentiment distributions, and trend forecasts
- ✅ **Data Processing**: MinMaxScaler normalization, technical indicators (MA, Returns, Volume)
- ✅ **Performance Optimization**: Caching with @st.cache_data and @st.cache_resource
- ✅ **100% Free**: No paid APIs, all open-source libraries

### User Workflow
1. **Enter Stock Symbol** (e.g., AAPL, TSLA, GOOGL, MSFT)
2. **Fetch Data** → Loads 1-year historical prices + generates news headlines
3. **View Analytics** → Price charts, sentiment distribution, news headlines
4. **Predict Trend** → AI trains LSTM model and forecasts future prices
5. **Analyze Results** → Compare predicted vs actual, view forecast with confidence metrics

## Project Architecture

### Directory Structure
```
financial_sentiment_stock_predictor/
├── app.py                          # Main Streamlit application
├── models/
│   ├── __init__.py
│   ├── sentiment_analyzer.py      # Financial sentiment analysis module
│   └── stock_predictor.py         # LSTM-based stock prediction model
├── utils/
│   ├── __init__.py
│   ├── text_processor.py          # Text cleaning and normalization
│   ├── data_preprocessor.py       # Data scaling and feature engineering
│   └── news_fetcher.py            # News headline generation
├── data/
│   └── sample_news.csv            # Sample financial news for fallback
└── .streamlit/
    └── config.toml                # Streamlit server configuration
```

### Technology Stack

#### Backend
- **Python 3.11**: Core programming language
- **TensorFlow/Keras**: LSTM neural network implementation
- **scikit-learn**: Data preprocessing (MinMaxScaler, metrics)
- **yfinance**: Free Yahoo Finance API for stock data
- **pandas/numpy**: Data manipulation and numerical operations

#### Frontend
- **Streamlit**: Web framework with reactive components
- **Plotly**: Interactive charts (candlestick, line, pie charts)
- **Custom CSS**: Gradient backgrounds and modern UI styling

#### ML Pipeline
- **Sentiment Analyzer**: Keyword-based financial sentiment scoring
- **LSTM Model**: 3-layer bidirectional LSTM with dropout (60-day lookback)
- **Feature Engineering**: Moving averages, price changes, volume analysis
- **Normalization**: MinMaxScaler for 0-1 range scaling

## Component Details

### 1. Sentiment Analyzer (`models/sentiment_analyzer.py`)
- **Type**: Custom keyword-based NLP
- **Keywords**: 30+ positive terms (surge, rally, profit) + 30+ negative terms (fall, loss, crisis)
- **Output**: Sentiment label (positive/negative/neutral) + score (-1 to +1)
- **Features**: Batch processing, aggregate sentiment calculation

### 2. LSTM Stock Predictor (`models/stock_predictor.py`)
- **Architecture**: Sequential model with 3 LSTM layers (50 units each)
- **Input**: 60-day sequences of OHLCV + technical indicators
- **Training**: Adam optimizer, MSE loss, early stopping (patience=5)
- **Predictions**: Historical backtest + future forecasting (5-30 days)
- **Metrics**: MAE, RMSE, R² score, accuracy percentage

### 3. Data Preprocessor (`utils/data_preprocessor.py`)
- **Feature Creation**: Returns, MA_5, MA_10, MA_20, High-Low Range, Volume Change
- **Normalization**: Separate scalers for features and target (Close price)
- **Sequence Creation**: Sliding window approach for LSTM input
- **Train/Test Split**: Configurable ratio (default 80/20)

### 4. Streamlit App (`app.py`)
- **UI Theme**: Gradient background (dark blue to purple)
- **State Management**: Session state for data persistence
- **Caching**: TTL-based caching for stock data (1 hour)
- **Responsive Design**: Mobile-friendly layout with collapsible sections
- **Error Handling**: Graceful fallbacks for API failures

## Performance Characteristics

### Speed
- **Data Fetch**: 2-5 seconds (depends on yfinance API)
- **Sentiment Analysis**: <1 second for 20 headlines
- **LSTM Training**: 30-90 seconds (30 epochs, early stopping)
- **Prediction Generation**: 1-3 seconds
- **Total Workflow**: ~60-120 seconds from fetch to predictions

### Resource Usage
- **CPU Only**: No GPU required (optimized for Replit)
- **Memory**: ~200-400 MB during LSTM training
- **Storage**: Minimal (no persistent model weights saved)

## Known Limitations & Future Enhancements

### Current Limitations
1. **Sentiment Source**: Uses generated sample headlines instead of real-time news API
2. **Model Persistence**: LSTM retrains on every prediction (no weight saving)
3. **Single Stock Analysis**: One ticker at a time

### Planned Enhancements (Next Phase)
1. ✨ **Multi-Stock Comparison**: Analyze multiple tickers simultaneously
2. 📊 **Advanced Indicators**: RSI, MACD, Bollinger Bands integration
3. 📄 **PDF Reports**: Downloadable prediction reports with charts
4. 🤖 **Reddit Integration**: Social media sentiment via PRAW
5. ⚙️ **Model Tuning**: UI for custom LSTM parameters

## Configuration

### Environment Variables
None required - 100% free operation

### Streamlit Configuration (`.streamlit/config.toml`)
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

### Workflow
- **Name**: Stock Predictor App
- **Command**: `streamlit run app.py --server.port 5000`
- **Status**: Running on port 5000

## Testing & Validation

### End-to-End Tests (Completed ✅)
- ✅ Data fetching for AAPL and TSLA
- ✅ Sentiment analysis generation and visualization
- ✅ LSTM training and prediction workflow
- ✅ All charts render correctly (candlestick, pie, line)
- ✅ Future forecast with trend direction
- ✅ Prediction metrics display (MAE, RMSE, R², Accuracy)

### Test Results
- **AAPL Test**: Passed - All features functional
- **TSLA Test**: Passed - Multi-ticker switching works
- **UI Responsiveness**: Passed - Elements load correctly
- **Error Handling**: Passed - Graceful fallbacks tested

## Usage Examples

### Supported Stock Symbols
- **US Stocks**: AAPL, TSLA, GOOGL, MSFT, AMZN, NVDA, META
- **Indian Stocks**: INFY.NS, TCS.NS, RELIANCE.NS
- **Any Yahoo Finance ticker**: Enter valid symbol

### Typical Workflow
```
1. Open app → Enter "AAPL" → Click "Fetch Data"
2. View price history, sentiment analysis, news headlines
3. Adjust forecast days slider (5-30 days)
4. Click "Predict Trend"
5. Review predicted vs actual comparison
6. Analyze future forecast and trend direction
7. Check prediction accuracy metrics
```

## Troubleshooting

### Common Issues
- **"Unable to fetch data"**: Check internet connection or try different ticker
- **Slow predictions**: Normal for first run (TensorFlow initialization)
- **Chart not displaying**: Refresh browser, check Plotly compatibility

### Debug Mode
- Logs available in workflow: `Stock Predictor App`
- TensorFlow warnings (GPU not found) are normal - CPU mode is intended

## Recent Changes

### October 28, 2025
- ✅ Initial project setup and architecture
- ✅ Implemented all core ML components
- ✅ Built complete Streamlit UI with gradient theme
- ✅ Added caching for performance optimization
- ✅ Completed end-to-end testing with multiple tickers
- ✅ Validated all MVP features functional

## Contributing & Development

### To Run Locally
```bash
# Already configured - workflow runs automatically
# Manual start: streamlit run app.py --server.port 5000
```

### To Modify
- **UI Changes**: Edit `app.py` (Streamlit components)
- **Model Architecture**: Adjust `models/stock_predictor.py`
- **Sentiment Logic**: Update `models/sentiment_analyzer.py`
- **Data Processing**: Modify `utils/data_preprocessor.py`

## License & Disclaimer

**License**: Open Source (All libraries used are free and open-source)

**⚠️ Important Disclaimer**: 
This is an educational project demonstrating machine learning applications in finance. 
**NOT FINANCIAL ADVICE**. Always conduct your own research before making investment decisions.
The predictions are for educational purposes only and should not be used as the sole basis for trading.

---

**Built with ❤️ using Python, Streamlit, TensorFlow & Scikit-learn**
**100% Free & Open Source** | **CPU-Only Optimized for Replit**

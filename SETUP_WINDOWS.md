# Windows Setup Guide for Stock Predictor App

## Prerequisites
- Python 3.11 or higher installed
- VS Code installed
- Internet connection

## Installation Steps

### 1. Open VS Code and Terminal
- Open VS Code
- Open Terminal (Ctrl + ` or View > Terminal)

### 2. Navigate to Project Directory
```bash
cd path\to\your\project\folder
```

### 3. Create Virtual Environment (Recommended)
```bash
python -m venv venv
```

### 4. Activate Virtual Environment
```bash
venv\Scripts\activate
```

### 5. Install Required Packages
```bash
pip install streamlit yfinance pandas numpy plotly scikit-learn tensorflow matplotlib requests beautifulsoup4
```

### 6. Run the Application
```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## Alternative: Run Without Virtual Environment
If you prefer not to use a virtual environment:

```bash
pip install streamlit yfinance pandas numpy plotly scikit-learn tensorflow matplotlib requests beautifulsoup4
streamlit run app.py
```

## Troubleshooting

### Error: "Python not found"
- Make sure Python is installed and added to PATH
- Download from: https://www.python.org/downloads/

### Error: "pip not recognized"
- Reinstall Python and check "Add to PATH" during installation

### Slow Installation
- TensorFlow is a large package (~500MB), installation may take 5-10 minutes

### Port Already in Use
If port 8501 is busy, Streamlit will automatically use the next available port.

## Project Structure
```
financial_sentiment_stock_predictor/
├── app.py                          # Main application file
├── models/
│   ├── sentiment_analyzer.py      # Sentiment analysis module
│   └── stock_predictor.py         # LSTM prediction model
├── utils/
│   ├── text_processor.py          # Text processing utilities
│   ├── data_preprocessor.py       # Data preprocessing
│   └── news_fetcher.py            # News fetching module
└── data/
    └── sample_news.csv            # Sample news data
```

## Usage
1. Enter a stock symbol (e.g., AAPL, TSLA, MSFT, GOOGL)
2. Click "Fetch Data" to load historical data
3. View stock analytics and sentiment analysis
4. Click "Predict Trend" to generate AI predictions
5. Analyze predicted vs actual prices and future forecasts

## Supported Stock Symbols
- US Stocks: AAPL, TSLA, GOOGL, MSFT, AMZN, NVDA, META
- Add `.NS` for Indian stocks: INFY.NS, TCS.NS, RELIANCE.NS
- Any valid Yahoo Finance ticker symbol

## Notes
- First run may take longer due to TensorFlow initialization
- LSTM training typically takes 30-90 seconds
- All data is fetched in real-time from Yahoo Finance (free)
- No API keys required

## Stopping the Application
- Press `Ctrl + C` in the terminal
- Close the browser tab

---

**100% Free & Open Source**

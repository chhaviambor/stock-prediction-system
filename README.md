# 📈 Financial Sentiment–Driven Stock Prediction System  

---

## 🧠 Overview  
This project is an AI-powered hybrid stock prediction system that integrates **historical stock data**, **financial news sentiment**, and **machine learning models** to predict stock price movements.  
The goal is to enhance accuracy and interpretability by combining **technical analysis** (past data trends) with **sentiment analysis** (market mood).  

---

## ⚙️ Features  
- 📰 **Financial News Sentiment Analysis:** Extracts and classifies sentiment (positive, negative, neutral) from financial news using NLP models.  
- 💹 **Stock Price Forecasting:** Uses historical data and deep learning models to predict future stock prices.  
- 🔗 **Hybrid Integration:** Combines sentiment and numerical data to improve prediction reliability.  
- 📊 **Interactive Dashboard:** Visualizes trends, sentiment polarity, and predictions through Streamlit.  
- 🧠 **Explainability:** Highlights how sentiment influences stock movements.  

---

## 🏗️ Architecture  
### The project consists of three major components:  
1. **Data Collection Module:**  
   - Fetches stock data via Yahoo Finance API.  
   - Scrapes financial news headlines from trusted sources.  

2. **Sentiment Analysis Module:**  
   - Uses pre-trained transformer models like **FinBERT** or **VADER** for text sentiment scoring.  

3. **Prediction Module:**  
   - Applies **LSTM/GRU-based neural networks** for time-series forecasting.  
   - Combines sentiment scores as additional features for improved performance.  

---

## 📁 Project Structure  
📂 stock-prediction-system/
│
├── app.py # Streamlit web interface
├── sentiment_analysis.py # NLP sentiment scoring logic
├── stock_prediction_model.py # LSTM/ML model for price prediction
├── data_preprocessing.py # Cleans and merges datasets
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── assets/ # Visuals, icons, and images

---

## 🧩 Technologies Used  

- **Programming Language:** Python  
- **Frameworks:** Streamlit, TensorFlow / Keras  
- **APIs:** Yahoo Finance API, NewsAPI  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, VADER / FinBERT  
- **Database:** CSV / SQLite (for local storage)  
- **Version Control:** Git & GitHub  

---

## 🚀 Installation and Setup  

Follow these steps to run the project locally:  

```bash
# 1️⃣ Clone the repository
git clone https://github.com/yourusername/stock-prediction-system.git
cd stock-prediction-system

# 2️⃣ Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # (Windows)
# or
source venv/bin/activate   # (Mac/Linux)

# 3️⃣ Install required dependencies
pip install -r requirements.txt

# 4️⃣ Run the Streamlit app
streamlit run app.py

##  🧮 How It Works 
Step 1 — Data Collection
Stock price data is fetched using Yahoo Finance API.
Financial news headlines are collected from NewsAPI or web scraping.

Step 2 — Sentiment Analysis
Each headline is processed through FinBERT or VADER to determine its sentiment polarity (positive, negative, neutral).

Step 3 — Feature Engineering
Sentiment scores are merged with stock indicators (moving average, RSI, etc.).

Step 4 — Model Training
The LSTM model learns from historical prices + sentiment-enhanced features to predict the next day’s stock price.

Step 5 — Visualization & Prediction
The user interface (Streamlit) displays:
Historical trends
Sentiment breakdown
Next-day forecast

📊 Sample Output
Sentiment Score: Positive (0.78)
Predicted Closing Price: ₹1450.32
Market Trend: Upward

📚 Future Scope
Integrate real-time live news sentiment updates.
Extend support for cryptocurrency and commodities.
Introduce portfolio recommendation system based on sentiment.

👩‍💻 Team & Contribution
Developer: Chhavi Ambor
Role: End-to-end design, implementation, and documentation
Year: Final Year B.Tech (Computer Engineering) — NMIMS Indore

🧾 License
This project is licensed under the MIT License — free to use and modify with attributio
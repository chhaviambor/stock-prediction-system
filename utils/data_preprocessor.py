"""
Data preprocessing utilities for stock prediction.
Handles normalization, scaling, and data preparation for LSTM models.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class StockDataPreprocessor:
    """
    Preprocessor for stock price and sentiment data.
    """
    
    def __init__(self):
        """Initialize the preprocessor with scalers."""
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
    
    def prepare_features(self, df, include_sentiment=True):
        """
        Prepare features from stock dataframe.
        
        Args:
            df (pd.DataFrame): Stock data with OHLCV columns
            include_sentiment (bool): Whether to include sentiment features
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        df = df.copy()
        
        # Calculate technical indicators
        df['Returns'] = df['Close'].pct_change()
        df['High_Low_Range'] = (df['High'] - df['Low']) / df['Low']
        df['Price_Change'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df
    
    def create_sequences(self, data, target, lookback=60):
        """
        Create sequences for LSTM model.
        
        Args:
            data (np.array): Feature data
            target (np.array): Target data
            lookback (int): Number of timesteps to look back
            
        Returns:
            tuple: (X, y) sequences
        """
        X, y = [], []
        
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def normalize_data(self, df, feature_columns, target_column='Close'):
        """
        Normalize features and target using MinMaxScaler.
        
        Args:
            df (pd.DataFrame): Dataframe with features
            feature_columns (list): List of feature column names
            target_column (str): Name of target column
            
        Returns:
            tuple: (normalized_features, normalized_target)
        """
        # Normalize features
        features = df[feature_columns].values
        normalized_features = self.feature_scaler.fit_transform(features)
        
        # Normalize target
        target = df[[target_column]].values
        normalized_target = self.price_scaler.fit_transform(target)
        
        self.is_fitted = True
        
        return normalized_features, normalized_target
    
    def inverse_transform_price(self, scaled_prices):
        """
        Convert scaled prices back to original scale.
        
        Args:
            scaled_prices (np.array): Scaled price values
            
        Returns:
            np.array: Original scale prices
        """
        if not self.is_fitted:
            return scaled_prices
        
        return self.price_scaler.inverse_transform(scaled_prices.reshape(-1, 1))
    
    def split_train_test(self, X, y, train_ratio=0.8):
        """
        Split data into training and testing sets.
        
        Args:
            X (np.array): Feature sequences
            y (np.array): Target values
            train_ratio (float): Ratio of training data
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * train_ratio)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test

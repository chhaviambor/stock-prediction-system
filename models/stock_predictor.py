"""
LSTM-based Stock Price Predictor.
Combines historical price data with sentiment analysis for prediction.
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from utils.data_preprocessor import StockDataPreprocessor

class LSTMStockPredictor:
    """
    LSTM model for stock price prediction.
    """
    
    def __init__(self, lookback=60, units=50):
        """
        Initialize the LSTM predictor.
        
        Args:
            lookback (int): Number of timesteps to look back
            units (int): Number of LSTM units
        """
        self.lookback = lookback
        self.units = units
        self.model = None
        self.preprocessor = StockDataPreprocessor()
        self.is_trained = False
    
    def build_model(self, input_shape):
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape (tuple): Shape of input (timesteps, features)
        """
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(units=self.units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(units=self.units, return_sequences=True),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(units=self.units, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        self.model = model
    
    def train(self, df, feature_columns, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the LSTM model.
        
        Args:
            df (pd.DataFrame): Training dataframe
            feature_columns (list): List of feature columns
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Validation data ratio
            
        Returns:
            dict: Training history
        """
        # Prepare features
        df_processed = self.preprocessor.prepare_features(df)
        
        # Normalize data
        normalized_features, normalized_target = self.preprocessor.normalize_data(
            df_processed, feature_columns
        )
        
        # Create sequences
        X, y = self.preprocessor.create_sequences(
            normalized_features, normalized_target, self.lookback
        )
        
        # Build model if not already built
        if self.model is None:
            self.build_model((X.shape[1], X.shape[2]))
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )
        
        self.is_trained = True
        return history.history
    
    def predict(self, df, feature_columns):
        """
        Make predictions on new data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_columns (list): List of feature columns
            
        Returns:
            np.array: Predicted prices
        """
        if not self.is_trained or self.model is None:
            return None
        
        # Prepare features
        df_processed = self.preprocessor.prepare_features(df)
        
        # Get the last lookback period
        features = df_processed[feature_columns].values
        
        if len(features) < self.lookback:
            return None
        
        # Normalize using the fitted scaler
        normalized_features = self.preprocessor.feature_scaler.transform(features)
        
        # Create sequences
        X, _ = self.preprocessor.create_sequences(
            normalized_features,
            np.zeros(len(normalized_features)),  # Dummy target
            self.lookback
        )
        
        # Predict
        predictions_scaled = self.model.predict(X, verbose=0)
        
        # Inverse transform to get actual prices
        predictions = self.preprocessor.inverse_transform_price(predictions_scaled)
        
        return predictions.flatten()
    
    def predict_future(self, df, feature_columns, days=10):
        """
        Predict future stock prices.
        
        Args:
            df (pd.DataFrame): Historical dataframe
            feature_columns (list): List of feature columns
            days (int): Number of days to predict
            
        Returns:
            np.array: Future price predictions
        """
        if not self.is_trained or self.model is None:
            return None
        
        # Prepare features
        df_processed = self.preprocessor.prepare_features(df)
        features = df_processed[feature_columns].values
        
        # Normalize
        normalized_features = self.preprocessor.feature_scaler.transform(features)
        
        # Get last sequence
        last_sequence = normalized_features[-self.lookback:]
        predictions = []
        
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, self.lookback, -1)
            
            # Predict next value
            next_pred_scaled = self.model.predict(X_pred, verbose=0)[0]
            
            # Create next feature vector (simplified - using last known features)
            next_features = current_sequence[-1].copy()
            # Update with predicted price (assuming Close is first feature)
            # This is a simplification; in practice, you'd update all features
            
            # Append prediction
            predictions.append(next_pred_scaled[0])
            
            # Update sequence
            current_sequence = np.vstack([current_sequence[1:], next_features])
        
        # Inverse transform predictions
        predictions_array = np.array(predictions).reshape(-1, 1)
        future_prices = self.preprocessor.inverse_transform_price(predictions_array)
        
        return future_prices.flatten()

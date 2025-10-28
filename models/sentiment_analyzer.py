"""
Financial Sentiment Analyzer using keyword-based approach.
This module analyzes financial news headlines and assigns sentiment scores.
"""
import pandas as pd
import numpy as np
from utils.text_processor import normalize_text

class FinancialSentimentAnalyzer:
    """
    Custom sentiment analyzer for financial text using keyword dictionaries.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer with financial keywords."""
        # Positive financial keywords
        self.positive_keywords = {
            'surge', 'rally', 'gains', 'profit', 'growth', 'increase', 'up', 'high',
            'bullish', 'boom', 'soar', 'jump', 'rise', 'advance', 'outperform',
            'beat', 'exceed', 'strong', 'positive', 'optimism', 'confidence',
            'record', 'breakthrough', 'innovation', 'success', 'boost', 'upgrade',
            'expansion', 'opportunity', 'momentum', 'recovery', 'rebound', 'improve'
        }
        
        # Negative financial keywords
        self.negative_keywords = {
            'fall', 'drop', 'decline', 'loss', 'decrease', 'down', 'low', 'bearish',
            'crash', 'plunge', 'tumble', 'sink', 'weak', 'negative', 'concern',
            'fear', 'risk', 'uncertainty', 'volatility', 'struggle', 'miss',
            'disappoint', 'cut', 'layoff', 'recession', 'crisis', 'trouble',
            'downgrade', 'warning', 'threat', 'pressure', 'challenge', 'disrupt'
        }
        
        # Neutral keywords (for context)
        self.neutral_keywords = {
            'announce', 'report', 'state', 'data', 'indicate', 'show', 'remain',
            'stable', 'unchanged', 'maintain', 'continue', 'steady'
        }
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a given text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary with sentiment label and score
        """
        if not text or not isinstance(text, str):
            return {'label': 'neutral', 'score': 0.0}
        
        # Normalize text
        normalized = normalize_text(text)
        words = normalized.split()
        
        # Count keyword occurrences
        positive_count = sum(1 for word in words if word in self.positive_keywords)
        negative_count = sum(1 for word in words if word in self.negative_keywords)
        
        # Calculate sentiment score (-1 to 1)
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment_score = 0.0
            label = 'neutral'
        else:
            sentiment_score = (positive_count - negative_count) / max(len(words), 1)
            
            # Normalize to -1 to 1 range
            sentiment_score = np.clip(sentiment_score * 5, -1, 1)
            
            if sentiment_score > 0.1:
                label = 'positive'
            elif sentiment_score < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
        
        return {
            'label': label,
            'score': float(sentiment_score),
            'positive_count': positive_count,
            'negative_count': negative_count
        }
    
    def analyze_batch(self, texts):
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            list: List of sentiment dictionaries
        """
        return [self.analyze_sentiment(text) for text in texts]
    
    def get_aggregate_sentiment(self, texts):
        """
        Get aggregate sentiment from multiple texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            dict: Aggregate sentiment statistics
        """
        sentiments = self.analyze_batch(texts)
        scores = [s['score'] for s in sentiments]
        labels = [s['label'] for s in sentiments]
        
        return {
            'average_score': np.mean(scores) if scores else 0.0,
            'positive_ratio': labels.count('positive') / len(labels) if labels else 0.0,
            'negative_ratio': labels.count('negative') / len(labels) if labels else 0.0,
            'neutral_ratio': labels.count('neutral') / len(labels) if labels else 0.0,
            'total_articles': len(texts)
        }

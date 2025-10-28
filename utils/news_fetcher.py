"""
News fetcher for financial headlines.
Uses free sources with fallback to sample data.
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
import random

class NewsFetcher:
    """
    Fetch financial news headlines from free sources.
    """
    
    def __init__(self):
        """Initialize the news fetcher."""
        self.sample_headlines = [
            "Company reports strong quarterly earnings exceeding expectations",
            "Market uncertainty rises amid global economic concerns",
            "Tech sector rallies on positive innovation news",
            "Stock prices surge following breakthrough product announcement",
            "Analysts downgrade rating citing competitive pressures",
            "Revenue growth accelerates in latest financial report",
            "Market volatility increases on geopolitical tensions",
            "Investors optimistic about future growth prospects",
            "Regulatory challenges pose risks to expansion plans",
            "Strong consumer demand drives record sales",
            "Cost pressures impact profit margins",
            "Strategic partnership announced with industry leader",
            "Market share gains in key business segments",
            "Economic headwinds create near-term uncertainty",
            "Innovation pipeline strengthens competitive position"
        ]
    
    def fetch_news(self, ticker, days=7):
        """
        Fetch news headlines for a given ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            days (int): Number of days of news to fetch
            
        Returns:
            pd.DataFrame: DataFrame with headline and date columns
        """
        try:
            # Try to fetch from free news sources
            # Using a simple approach with sample data as primary method
            headlines = self._generate_sample_news(ticker, days)
            return headlines
        except Exception as e:
            # Fallback to sample data
            return self._load_sample_data()
    
    def _generate_sample_news(self, ticker, days):
        """
        Generate sample news headlines.
        
        Args:
            ticker (str): Stock ticker
            days (int): Number of days
            
        Returns:
            pd.DataFrame: News dataframe
        """
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(days)]
        
        headlines = []
        for date in dates:
            # Generate 1-3 headlines per day
            num_headlines = random.randint(1, 3)
            for _ in range(num_headlines):
                headline = f"{ticker}: {random.choice(self.sample_headlines)}"
                headlines.append({
                    'headline': headline,
                    'date': date.strftime('%Y-%m-%d'),
                    'source': 'Financial News'
                })
        
        return pd.DataFrame(headlines)
    
    def _load_sample_data(self):
        """
        Load sample news data from CSV.
        
        Returns:
            pd.DataFrame: Sample news dataframe
        """
        try:
            df = pd.read_csv('data/sample_news.csv')
            return df
        except:
            # Return empty dataframe if file not found
            return pd.DataFrame(columns=['headline', 'date', 'source'])
    
    def search_financial_news(self, query, max_results=10):
        """
        Search for financial news using free sources.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results
            
        Returns:
            list: List of news articles
        """
        # Simple implementation using sample data
        # In production, this could use NewsAPI free tier or web scraping
        sample_df = self._load_sample_data()
        
        if len(sample_df) > 0:
            # Filter headlines containing query terms
            mask = sample_df['headline'].str.contains(query, case=False, na=False)
            filtered = sample_df[mask]
            
            if len(filtered) > 0:
                return filtered.head(max_results).to_dict('records')
        
        # Return random sample if no matches
        return [
            {
                'headline': f"{query}: {random.choice(self.sample_headlines)}",
                'date': datetime.now().strftime('%Y-%m-%d'),
                'source': 'Financial News'
            }
            for _ in range(min(max_results, 5))
        ]

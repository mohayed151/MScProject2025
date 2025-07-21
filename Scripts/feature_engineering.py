import pandas as pd
import numpy as np
import ta  # Technical Analysis library
import tweepy
import logging
from textblob import TextBlob
from datetime import datetime, timedelta
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_technical_indicators(df):
    """
    Calculate technical indicators (RSI, MACD, Bollinger Bands, VWAP) for intra-day data.
    
    Args:
        df (pd.DataFrame): Intra-day data with Open, High, Low, Close, Volume columns.
    
    Returns:
        pd.DataFrame: DataFrame with added indicator columns.
    """
    try:
        logger.info("Calculating technical indicators")
        df = df.copy()
        
        # Relative Strength Index (RSI, 14-period)
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # Moving Average Convergence Divergence (MACD)
        macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # Bollinger Bands (20-period, 2 std)
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        
        # Volume Weighted Average Price (VWAP)
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Drop rows with NaN values from indicators
        df.dropna(inplace=True)
        
        return df
    
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        raise

def calculate_sentiment_scores(ticker='^GSPC', api_key=None, api_secret=None, access_token=None, access_secret=None):
    """
    Calculate sentiment scores from Twitter data using Tweepy and TextBlob.
    
    Args:
        ticker (str): Ticker symbol for sentiment analysis.
        api_key, api_secret, access_token, access_secret (str): Twitter API credentials.
    
    Returns:
        pd.DataFrame: Sentiment scores aligned with intra-day data timestamps.
    """
    try:
        logger.info(f"Fetching Twitter sentiment for {ticker}")
        
        # Initialize Twitter API (replace with your credentials)
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)
        
        # Fetch tweets from the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        query = f"${ticker} -filter:retweets"
        tweets = tweepy.Cursor(api.search_tweets, q=query, lang='en', 
                              since=start_date.strftime('%Y-%m-%d'), 
                              until=end_date.strftime('%Y-%m-%d')).items(1000)
        
        # Calculate sentiment scores
        sentiment_data = []
        for tweet in tweets:
            text = tweet.text
            sentiment = TextBlob(text).sentiment.polarity
            timestamp = tweet.created_at
            sentiment_data.append({'Timestamp': timestamp, 'Sentiment': sentiment})
        
        sentiment_df = pd.DataFrame(sentiment_data)
        sentiment_df['Timestamp'] = pd.to_datetime(sentiment_df['Timestamp']).dt.tz_localize('UTC')
        
        # Aggregate sentiment by minute
        sentiment_df = sentiment_df.groupby(pd.Grouper(key='Timestamp', freq='1min')).mean().reset_index()
        
        return sentiment_df
    
    except Exception as e:
        logger.error(f"Error calculating sentiment scores: {str(e)}")
        raise

def main():
    """
    Main function to load intra-day data, calculate features, and save to CSV.
    """
    try:
        # Load intra-day data
        input_path = 'data/intraday/^GSPC_intraday.csv'
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} not found.")
        
        df = pd.read_csv(input_path, index_col='Datetime', parse_dates=True)
        logger.info(f"Loaded intra-day data with shape: {df.shape}")
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Calculate sentiment scores (replace with your Twitter API credentials)
        sentiment_df = calculate_sentiment_scores(
            api_key='YOUR_API_KEY',
            api_secret='YOUR_API_SECRET',
            access_token='YOUR_ACCESS_TOKEN',
            access_secret='YOUR_ACCESS_SECRET'
        )
        
        # Merge sentiment with price data
        df = df.merge(sentiment_df, left_index=True, right_on='Timestamp', how='left')
        df['Sentiment'].fillna(0, inplace=True)  # Neutral sentiment for missing data
        
        # Save features
        os.makedirs('data/features', exist_ok=True)
        output_path = 'data/features/sp500_intraday_features.csv'
        df.to_csv(output_path)
        logger.info(f"Features saved to {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    features_df = main()
    print("Features data head:\n", features_df.head())

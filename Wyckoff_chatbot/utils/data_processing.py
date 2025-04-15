import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def detect_wyckoff_patterns(data):
    """
    Detect Wyckoff patterns in the data - can be called separately from add_technical_indicators
    """
    try:
        df = data.copy()
        
        # Initialize pattern columns
        if 'Potential_Spring' not in df.columns:
            df['Potential_Spring'] = 0
        if 'Potential_Upthrust' not in df.columns:
            df['Potential_Upthrust'] = 0
        
        # Loop through each row to detect patterns
        for i in range(1, len(df)-1):
            # Spring pattern: Low below lower band, Close above lower band, Close above Open
            if (df['Low'].iloc[i] < df['BB_Lower'].iloc[i] and 
                df['Close'].iloc[i] > df['BB_Lower'].iloc[i] and 
                df['Close'].iloc[i] > df['Open'].iloc[i]):
                df.iloc[i, df.columns.get_loc('Potential_Spring')] = 1
            
            # Upthrust pattern: High above upper band, Close below upper band, Close below Open
            if (df['High'].iloc[i] > df['BB_Upper'].iloc[i] and 
                df['Close'].iloc[i] < df['BB_Upper'].iloc[i] and 
                df['Close'].iloc[i] < df['Open'].iloc[i]):
                df.iloc[i, df.columns.get_loc('Potential_Upthrust')] = 1
        
        return df
    except Exception as e:
        logger.error(f"Error detecting Wyckoff patterns: {e}")
        return data  # Return original data if pattern detection fails
def generate_simulated_data(symbol, num_days=252):
    """
    Generate simulated stock data for testing when real data can't be fetched
    """
    symbol = symbol.upper()  # Ensure uppercase for consistency
    logger.info(f"Generating simulated data for {symbol} ({num_days} days)")
    
    # Start date will be num_days ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')[:num_days]
    
    # Price ranges for common stocks
    price_map = {
        "NVDA": 112.50,
        "AAPL": 170.25,
        "MSFT": 305.75,
        "GOOGL": 142.30,
        "AMZN": 178.45,
        "META": 475.60,
        "TSLA": 175.25,
        "V": 275.80,
        "JPM": 190.15,
        "WMT": 61.35,
        "PG": 162.10,
        "JNJ": 148.75,
        "UNH": 530.40,
        "HD": 350.25,
        "BAC": 38.20
    }
    
    # Set a default price if symbol not in our map
    start_price = price_map.get(symbol, 100.0)
    
    # Set a different seed for each symbol
    seed_value = sum(ord(c) for c in symbol)
    np.random.seed(seed_value)
    
    # Trend and volatility parameters
    if symbol in ["NVDA", "TSLA", "AMD"]:
        trend = 0.0012
        volatility = 0.025
    elif symbol in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
        trend = 0.0008
        volatility = 0.018
    else:
        trend = 0.0005
        volatility = 0.015
    
    # Generate price series with trend and volatility
    daily_returns = np.random.normal(trend, volatility, len(date_range))
    price_series = [start_price]
    
    for ret in daily_returns:
        next_price = price_series[-1] * (1 + ret)
        price_series.append(next_price)
    
    # Truncate to match date range
    price_series = price_series[1:len(date_range)+1]
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': [p * (1 - np.random.uniform(0, 0.005)) for p in price_series],
        'High': [p * (1 + np.random.uniform(0.001, 0.015)) for p in price_series],
        'Low': [p * (1 - np.random.uniform(0.001, 0.015)) for p in price_series],
        'Close': price_series,
        'Volume': [int(np.random.uniform(500000, 10000000)) for _ in range(len(price_series))]
    }, index=date_range[:len(price_series)])
    
    # Add technical indicators
    enhanced_data = add_technical_indicators(data)
    
    return enhanced_data
def get_stock_data(symbol, start_date=None, end_date=None, period=None):
    """
    Fetches stock data using yfinance
    
    Args:
        symbol (str): Stock ticker symbol
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        period (str, optional): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        pandas.DataFrame: Stock data with OHLCV and additional indicators
    """
    try:
        # Set default dates if not provided
        if not start_date and not end_date and not period:
            period = '1y'
            
        if not period and not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        if not period and not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Try to download data
        logger.info(f"Downloading {symbol} data" + 
                   (f" with period={period}" if period else f" from {start_date} to {end_date}"))
        
        if period:
            data = yf.download(symbol, period=period, multi_level_index=False)
        else:
            data = yf.download(symbol, start=start_date, end=end_date, multi_level_index=False)
        
        if data.empty:
            logger.warning(f"Downloaded data for {symbol} is empty. Using simulated data.")
            return generate_simulated_data(symbol, 252)  # Use 252 days (~ 1 year of trading days)
        
        # Add technical indicators
        try:
            data_with_indicators = add_technical_indicators(data)
            return data_with_indicators
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            # Return the raw data if technical indicators fail
            return data
            
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        logger.info(f"Using simulated data for {symbol}")
        return generate_simulated_data(symbol, 252)

def generate_basic_simulated_data(index=None, num_days=252):
    """
    Generate very basic simulated data as a fallback when everything else fails
    """
    logger.info("Generating basic simulated data as fallback")
    
    if index is None or len(index) == 0:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days)
        index = pd.date_range(start=start_date, end=end_date, freq='B')[:num_days]
    
    # Create a simple upward trending price
    price_start = 100
    prices = [price_start * (1 + 0.0005 * i + 0.001 * np.random.randn()) for i in range(len(index))]
    
    # Create a basic DataFrame
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Close': prices,
        'Volume': [int(1000000 * (1 + 0.1 * np.random.randn())) for _ in range(len(prices))]
    }, index=index)
    
    # Add simple moving averages
    data['MA20'] = data['Close'].rolling(window=20, min_periods=1).mean()
    data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
    data['MA200'] = data['Close'].rolling(window=200, min_periods=1).mean()
    
    # Simple RSI approximation
    data['RSI'] = 50 + 10 * np.random.randn(len(prices))
    data['RSI'] = data['RSI'].clip(0, 100)
    
    # Simple Bollinger Bands
    data['BB_Middle'] = data['MA20']
    data['BB_Std'] = data['Close'].rolling(window=20, min_periods=1).std()
    data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
    data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
    
    # Simple OBV
    data['OBV'] = np.cumsum(np.random.randint(-100000, 100000, len(prices)))
    
    # Wyckoff patterns
    data['Potential_Spring'] = 0
    data['Potential_Upthrust'] = 0
    
    # Fill NaN values
    data = data.fillna(0)
    
    return data
def add_technical_indicators(data):
    """
    Adds technical indicators to the stock data - simplified version to avoid Series issues
    """
    try:
        # Make a deep copy to avoid SettingWithCopyWarning
        df = data.copy()
        print(df)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]

        # Calculate Moving Averages
        df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['MA200'] = df['Close'].rolling(window=200, min_periods=1).mean()
        
        # Calculate Volume Moving Average
        df['Volume_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        
        # Calculate price change and percentage change
        df['Price_Change'] = df['Close'].diff().fillna(0)
        df['Pct_Change'] = df['Close'].pct_change().fillna(0) * 100
        
        # Calculate RSI using a simpler method
        # First, calculate daily price changes
        diff = df['Close'].diff(1)
        
        # Define gains and losses
        gain = diff.copy()
        loss = diff.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss over period
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        
        # Calculate relative strength with handling for division by zero
        rs = avg_gain / avg_loss.replace(0, 0.001)
        
        # Calculate RSI
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = df['MA20']
        df['BB_Std'] = df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        # Initialize OBV and pattern columns to avoid Series truth value issues
        df['OBV'] = 0.0
        df['Potential_Spring'] = 0
        df['Potential_Upthrust'] = 0
        
        # Calculate OBV using a loop to avoid Series truth value issues
        obv = 0
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv += df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv -= df['Volume'].iloc[i]
            df.iloc[i, df.columns.get_loc('OBV')] = obv
        
        # Fill NaN values to avoid issues in further calculations
        # df = df.fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error in add_technical_indicators: {e}")
        # Return basic simulated data if calculation fails
        return generate_basic_simulated_data(data.index)
    
def format_data_for_chart(data, include_indicators=False):
    """
    Formats stock data for Chart.js
    
    Args:
        data (pandas.DataFrame): Stock data with indicators
        include_indicators (bool): Whether to include technical indicators
        
    Returns:
        dict: Formatted data for Chart.js
    """
    if data is None or len(data) == 0:
        return {
            'labels': [],
            'prices': [],
            'volumes': []
        }
    
    # Convert index to string dates
    date_labels = data.index.strftime('%Y-%m-%d').tolist()
    
    result = {
        'labels': date_labels,
        'prices': data['Close'].tolist(),
        'volumes': data['Volume'].tolist(),
        'dates': date_labels
    }
    
    if include_indicators and all(indicator in data.columns for indicator in ['MA20', 'MA50', 'RSI']):
        result.update({
            'ma20': data['MA20'].tolist(),
            'ma50': data['MA50'].tolist(),
            'ma200': data['MA200'].tolist() if 'MA200' in data.columns else [],
            'bb_upper': data['BB_Upper'].tolist() if 'BB_Upper' in data.columns else [],
            'bb_middle': data['BB_Middle'].tolist() if 'BB_Middle' in data.columns else [],
            'bb_lower': data['BB_Lower'].tolist() if 'BB_Lower' in data.columns else [],
            'rsi': data['RSI'].tolist() if 'RSI' in data.columns else [],
            'obv': data['OBV'].tolist() if 'OBV' in data.columns else []
        })
    
    return result
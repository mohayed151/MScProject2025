"""
Debug script to understand the data structure issue in feature engineering
"""
import pandas as pd
import numpy as np
from pathlib import Path
from config import *

def debug_data_structure():
    """Debug the data structure to identify the aggregation issue."""
    
    # Load the raw data file
    data_file = RAW_DATA_DIR / f"combined_ohlcv_prices_{INTERVAL}.csv"
    print(f"Loading data from: {data_file}")
    
    if not data_file.exists():
        print("Data file does not exist!")
        return
    
    # Try loading with different approaches
    print("\n=== Method 1: Load as saved ===")
    try:
        df1 = pd.read_csv(data_file, header=[0,1], index_col=0, parse_dates=True)
        print(f"Shape: {df1.shape}")
        print(f"Index type: {type(df1.index)}")
        print(f"Column structure:")
        print(df1.columns)
        print(f"Column levels: {df1.columns.nlevels}")
        print(f"Data types:\n{df1.dtypes}")
        print(f"First few rows:")
        print(df1.head())
        
        # Try to access single ticker data
        print(f"\n=== Trying to access ^GSPC data ===")
        if '^GSPC' in df1.columns.get_level_values(0):
            gspc_data = df1['^GSPC']
            print(f"^GSPC data shape: {gspc_data.shape}")
            print(f"^GSPC data types:\n{gspc_data.dtypes}")
            print(f"^GSPC data head:")
            print(gspc_data.head())
            
            # Check for numeric columns
            numeric_cols = gspc_data.select_dtypes(include=[np.number]).columns
            print(f"Numeric columns: {list(numeric_cols)}")
            
            # Try a simple rolling operation
            try:
                test_rolling = gspc_data['Close'].rolling(5).mean()
                print("Rolling operation successful!")
            except Exception as e:
                print(f"Rolling operation failed: {e}")
                
        else:
            print("^GSPC not found in column level 0")
            print(f"Available tickers: {df1.columns.get_level_values(0).unique()}")
            
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    print("\n=== Method 2: Load without multi-index ===")
    try:
        df2 = pd.read_csv(data_file, index_col=0, parse_dates=True)
        print(f"Shape: {df2.shape}")
        print(f"Columns: {list(df2.columns)}")
        print(f"Data types:\n{df2.dtypes}")
        print(f"First few rows:")
        print(df2.head())
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    print("\n=== Method 3: Raw inspection ===")
    try:
        with open(data_file, 'r') as f:
            lines = f.readlines()
            print("First 10 lines of the file:")
            for i, line in enumerate(lines[:10]):
                print(f"Line {i}: {line.strip()}")
    except Exception as e:
        print(f"Method 3 failed: {e}")

if __name__ == "__main__":
    debug_data_structure()
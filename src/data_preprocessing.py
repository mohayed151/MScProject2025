import os
import pandas as pd
import requests
import ta
from datetime import datetime, timedelta

API_KEY = "VgIZ5L_tjJyzRJkR8T31z1X4jFqWiDcU"
DATA_FILE = "data/SPY_processed.csv"

def get_polygon_chunk(symbol, start_date, end_date, multiplier=5, timespan="minute"):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df

def get_polygon_data(symbol="SPY", years=2):
    print(f"[INFO] Fetching {years} years of {symbol} data from Polygon.io...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)

    all_data = []
    chunk_size_days = 180  # 6 months per request
    chunk_start = start_date

    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=chunk_size_days), end_date)
        print(f"[INFO] Fetching chunk {chunk_start.date()} to {chunk_end.date()}...")
        df_chunk = get_polygon_chunk(symbol, chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        if not df_chunk.empty:
            all_data.append(df_chunk)
        chunk_start = chunk_end + timedelta(days=1)

    df_all = pd.concat(all_data).drop_duplicates().sort_values("timestamp")
    df_all.set_index("timestamp", inplace=True)
    print(f"[INFO] Total bars retrieved: {len(df_all)}")
    return df_all

def add_technical_indicators(df):
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    df["ema_10"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator()
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_bbh"] - df["bb_bbl"]
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(window=20).std()
    df.dropna(inplace=True)
    return df

def save_data(df):
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_FILE)
    print(f"[INFO] Data saved to {DATA_FILE}")

if __name__ == "__main__":
    if os.path.exists(DATA_FILE):
        print(f"[INFO] Processed SPY data already exists at {DATA_FILE}. Skipping download.")
    else:
        raw_df = get_polygon_data("SPY", years=2)
        processed_df = add_technical_indicators(raw_df)
        save_data(processed_df)
    print("[DONE] SPY Polygon.io data processing complete.")

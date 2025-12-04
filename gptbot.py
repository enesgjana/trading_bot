import pandas as pd
import numpy as np
import ccxt
import time
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================
# CONFIGURATION
# ============================================
class Config:
    SYMBOL = "BTC/USDT"
    TIMEFRAME = "1h"
    DATA_POINTS = 300
    MODEL_PATH = Path("model.pkl")
    SCALER_PATH = Path("scaler.pkl")
    INITIAL_BALANCE = 10000.0
    POSITION_SIZE = 0.2
    RETRAIN_INTERVAL_HOURS = 24
    FETCH_INTERVAL_SECONDS = 10  # 1 hour
    LOG_FILE = "bot.log"

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    filename=Config.LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ============================================
# EXCHANGE INITIALIZATION
# ============================================
def init_exchange():
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })
    return exchange

# ============================================
# FETCH DATA
# ============================================
def fetch_data(exchange):
    ohlcv = exchange.fetch_ohlcv(Config.SYMBOL, Config.TIMEFRAME, limit=Config.DATA_POINTS)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# ============================================
# FEATURE ENGINEERING
# ============================================
def add_indicators(df):
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_30"] = df["close"].rolling(30).mean()
    df["rsi"] = compute_rsi(df["close"], 14)
    df["returns"] = df["close"].pct_change()
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ============================================
# LABELING
# ============================================
def label_data(df):
    df["future_close"] = df["close"].shift(-1)
    df["signal"] = np.where(df["future_close"] > df["close"], 1, 0)
    df.dropna(inplace=True)
    return df

# ============================================
# MODEL TRAINING
# ============================================
def train_model(df):
    features = ["sma_10", "sma_30", "rsi", "returns"]
    X = df[features]
    y = df["signal"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump(model, Config.MODEL_PATH)
    joblib.dump(scaler, Config.SCALER_PATH)

    logging.info("✅ Model trained and saved.")
    return model, scaler

# ============================================
# LOAD MODEL
# ============================================
def load_model():
    if Config.MODEL_PATH.exists() and Config.SCALER_PATH.exists():
        model = joblib.load(Config.MODEL_PATH)
        scaler = joblib.load(Config.SCALER_PATH)
        logging.info("📦 Model and scaler loaded.")
        return model, scaler
    else:
        logging.warning("⚠️ No model found. Training new one...")
        return None, None

# ============================================
# TRADING SIMULATION
# ============================================
class Portfolio:
    def __init__(self, balance):
        self.balance = balance
        self.position = 0.0
        self.entry_price = 0.0
        self.pnl = 0.0
        self.trades = []  # store trades for chart

    def buy(self, price, fraction, timestamp):
        if self.position == 0:
            size = self.balance * fraction / price
            self.balance -= size * price
            self.position = size
            self.entry_price = price
            self.trades.append((timestamp, price, "buy"))
            logging.info(f"🟢 Bought {size:.5f} BTC at {price:.2f}")

    def sell(self, price, timestamp):
        if self.position > 0:
            proceeds = self.position * price
            profit = proceeds - (self.position * self.entry_price)
            self.balance += proceeds
            self.pnl += profit
            self.trades.append((timestamp, price, "sell"))
            logging.info(f"🔴 Sold {self.position:.5f} BTC at {price:.2f} | Profit: {profit:.2f}")
            self.position = 0.0

# ============================================
# CHART VISUALIZATION
# ============================================
def plot_trades(df, trades):
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["close"], label="Close Price", color="blue")

    for ts, price, action in trades:
        if action == "buy":
            plt.scatter(ts, price, color="green", label="Buy Signal", marker="^", s=100)
        elif action == "sell":
            plt.scatter(ts, price, color="red", label="Sell Signal", marker="v", s=100)

    plt.title("BTC/USDT Trading Simulation")
    plt.xlabel("Time")
    plt.ylabel("Price (USDT)")
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================================
# MAIN TRADING LOOP
# ============================================
def main():
    exchange = init_exchange()
    model, scaler = load_model()
    portfolio = Portfolio(Config.INITIAL_BALANCE)
    last_train_time = datetime.utcnow() - timedelta(hours=Config.RETRAIN_INTERVAL_HOURS)

    while True:
        try:
            df = fetch_data(exchange)
            df = add_indicators(df)
            df = label_data(df)

            if datetime.utcnow() - last_train_time > timedelta(hours=Config.RETRAIN_INTERVAL_HOURS):
                model, scaler = train_model(df)
                last_train_time = datetime.utcnow()

            features = ["sma_10", "sma_30", "rsi", "returns"]
            X_latest = scaler.transform([df[features].iloc[-1]])
            pred = model.predict(X_latest)[0]
            prob = model.predict_proba(X_latest)[0][pred]

            price = df["close"].iloc[-1]
            timestamp = df["timestamp"].iloc[-1]

            # TRADING LOGIC
            if pred == 1 and prob > 0.6:
                portfolio.buy(price, Config.POSITION_SIZE, timestamp)
            elif pred == 0 and portfolio.position > 0:
                portfolio.sell(price, timestamp)

            logging.info(f"💰 Balance: {portfolio.balance:.2f} | PnL: {portfolio.pnl:.2f}")
            print(f"[{datetime.now()}] Balance: {portfolio.balance:.2f} | PnL: {portfolio.pnl:.2f}")

            # Show chart after each iteration
            plot_trades(df, portfolio.trades)

        except Exception as e:
            logging.error(f"❌ Error in loop: {e}")
            print(f"❌ Error: {e}")

        time.sleep(Config.FETCH_INTERVAL_SECONDS)

# ============================================
if __name__ == "__main__":
    main()

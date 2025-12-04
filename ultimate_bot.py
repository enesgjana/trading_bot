import ccxt
import pandas as pd
import numpy as np
import os
import time
import csv
from datetime import datetime
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 🎯 CONFIGURATION
# ===============================
class TradingConfig:
    # Trading mode
    LIVE_TRADING = False
    
    # Capital management
    INITIAL_CAPITAL = 10000.0
    MIN_CAPITAL = 2000.0
    MAX_DAILY_LOSS = 1500.0
    MAX_TOTAL_LOSS_PCT = 3000.0
    
    # Risk management
    BASE_STOP_LOSS_PCT = 500.0
    BASE_TAKE_PROFIT_PCT = 1000.0
    MAX_POSITION_PCT = 0.45  # 45% of capital per trade
    MIN_CONFIDENCE = 0.2     # LOWERED from 0.6 to 0.45 - was too strict!
    
    # Trading limits
    MAX_TRADES_PER_DAY = 20
    MIN_ORDER_SIZE = 10.0
    TRADE_FEE_PCT = 0.1
    
    # Timing
    CHECK_INTERVAL = 5 * 60
    RETRAIN_INTERVAL = 7 * 24 * 60 * 60
    
    # Model settings
    USE_ENSEMBLE = True
    USE_KELLY_SIZING = True
    USE_REGIME_DETECTION = True
    
    # Files
    STATE_FILE = "trading_state.json"
    LOG_FILE = "trades_log.csv"
    METRICS_FILE = "performance_metrics.json"

# ===============================
# 📊 ADVANCED TECHNICAL INDICATORS
# ===============================
class AdvancedIndicators:
    @staticmethod
    def calculate_all(df):
        """Calculate 30+ advanced indicators"""
        df = df.copy()
        
        # Basic moving averages
        for period in [10, 20, 50, 100, 200]:
            df[f'SMA{period}'] = df['close'].rolling(period, min_periods=period).mean()
            df[f'EMA{period}'] = df['close'].ewm(span=period, adjust=False, min_periods=period).mean()
        
        # MACD with signal
        df['MACD'] = df['EMA10'] - df['EMA50']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20, min_periods=20).mean()
        std20 = df['close'].rolling(20, min_periods=20).std()
        df['BB_upper'] = sma20 + 2*std20
        df['BB_middle'] = sma20
        df['BB_lower'] = sma20 - 2*std20
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-10)
        
        # RSI with oversold/overbought
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].clip(0, 100)
        df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
        df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
        
        # Stochastic Oscillator
        low14 = df['low'].rolling(14, min_periods=14).min()
        high14 = df['high'].rolling(14, min_periods=14).max()
        df['Stoch_K'] = 100 * (df['close'] - low14) / ((high14 - low14) + 1e-10)
        df['Stoch_K'] = df['Stoch_K'].clip(0, 100)
        df['Stoch_D'] = df['Stoch_K'].rolling(3, min_periods=3).mean()
        
        # ATR and volatility
        df['H-L'] = df['high'] - df['low']
        df['H-Cp'] = abs(df['high'] - df['close'].shift(1))
        df['L-Cp'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['H-L','H-Cp','L-Cp']].max(axis=1)
        df['ATR'] = df['TR'].rolling(14, min_periods=14).mean()
        df['ATR_pct'] = df['ATR'] / df['close']
        
        # Volatility measures
        df['Volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['Volatility_ratio'] = df['Volatility'] / df['Volatility'].rolling(50).mean()
        
        # Momentum indicators
        df['ROC'] = df['close'].pct_change(periods=12) * 100
        df['Momentum'] = df['close'].diff(10)
        
        # Volume indicators
        df['Volume_SMA'] = df['volume'].rolling(20).mean()
        df['Volume_ratio'] = df['volume'] / (df['Volume_SMA'] + 1e-10)
        df['OBV'] = (df['volume'] * (~df['close'].diff().le(0) * 2 - 1)).cumsum()
        df['OBV_EMA'] = df['OBV'].ewm(span=20, adjust=False).mean()
        
        # Price patterns
        df['Higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['Lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['Body_size'] = abs(df['close'] - df['open']) / df['open']
        df['Upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
        df['Lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']
        
        # Trend strength (ADX)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr_smooth = df['TR'].rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr_smooth)
        minus_di = 100 * (minus_dm.rolling(14).sum() / tr_smooth)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['ADX'] = dx.rolling(14).mean()
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di
        
        # CCI (Commodity Channel Index)
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: abs(x - x.mean()).mean())
        df['CCI'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        df['CCI'] = df['CCI'].clip(-300, 300)
        
        # Williams %R
        high14 = df['high'].rolling(14).max()
        low14 = df['low'].rolling(14).min()
        df['Williams_R'] = -100 * (high14 - df['close']) / (high14 - low14 + 1e-10)
        
        # Returns at multiple horizons
        df['Return_1'] = df['close'].pct_change(1)
        df['Return_5'] = df['close'].pct_change(5)
        df['Return_10'] = df['close'].pct_change(10)
        
        # Price distance from moving averages
        df['Distance_SMA20'] = (df['close'] - df['SMA20']) / df['SMA20']
        df['Distance_SMA50'] = (df['close'] - df['SMA50']) / df['SMA50']
        df['Distance_SMA200'] = (df['close'] - df['SMA200']) / df['SMA200']
        
        # Moving average crossovers
        df['SMA_cross_20_50'] = ((df['SMA20'] > df['SMA50']).astype(int) - 
                                 (df['SMA20'] < df['SMA50']).astype(int))
        df['EMA_cross_10_50'] = ((df['EMA10'] > df['EMA50']).astype(int) - 
                                 (df['EMA10'] < df['EMA50']).astype(int))
        
        return df
    
    @staticmethod
    def add_regime_features(df):
        """Detect market regime"""
        
        # Trend detection
        df['Trending'] = (df['ADX'] > 25).astype(int)
        df['Strong_trend'] = (df['ADX'] > 40).astype(int)
        
        # Volatility regime
        vol_percentile = df['Volatility'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        df['High_volatility'] = (vol_percentile > 0.75).astype(int)
        df['Low_volatility'] = (vol_percentile < 0.25).astype(int)
        
        # Market direction
        df['Bull_market'] = ((df['close'] > df['SMA50']) & 
                            (df['SMA50'] > df['SMA200'])).astype(int)
        df['Bear_market'] = ((df['close'] < df['SMA50']) & 
                            (df['SMA50'] < df['SMA200'])).astype(int)
        
        # Range position
        range_50_high = df['high'].rolling(50).max()
        range_50_low = df['low'].rolling(50).min()
        df['Range_position'] = (df['close'] - range_50_low) / (range_50_high - range_50_low + 1e-10)
        
        return df

# ===============================
# 🎓 SMART LABELING - FIXED
# ===============================
class SmartLabeling:
    @staticmethod
    def adaptive_threshold(df, base_threshold=0.002):  # Increased from 0.001 to 0.002
        """Dynamic labeling based on market volatility - FIXED to create more signals"""
        
        # Threshold adapts to ATR - MORE LENIENT
        df['Threshold'] = df['ATR_pct'] * 1.0  # Reduced from 1.5 to 1.0
        df['Threshold'] = df['Threshold'].clip(base_threshold, 0.03)  # Reduced max from 0.05 to 0.03
        
        # Multi-period returns
        df['Future_return_1'] = df['close'].shift(-1) / df['close'] - 1
        df['Future_return_4'] = df['close'].shift(-4) / df['close'] - 1
        df['Future_return_24'] = df['close'].shift(-24) / df['close'] - 1
        
        # Voting across timeframes - MORE LENIENT
        buy_votes = 0
        sell_votes = 0
        
        for period in [1, 4, 24]:
            col = f'Future_return_{period}'
            if col in df.columns:
                buy_votes += (df[col] > df['Threshold']).astype(int)
                sell_votes += (df[col] < -df['Threshold']).astype(int)
        
        # FIXED: Only require 1+ timeframe agreement instead of 2+
        df['signal'] = np.where(buy_votes >= 1, 1,
                       np.where(sell_votes >= 1, -1, 0))
        
        # Quality filter - MORE LENIENT
        df['signal_strength'] = np.maximum(buy_votes, sell_votes) / 3
        df.loc[df['signal_strength'] < 0.33, 'signal'] = 0  # Reduced from 0.6 to 0.33
        
        return df

# ===============================
# 🤖 ENSEMBLE MODEL - FIXED
# ===============================
class EnsembleModel:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.label_mapping = None  # Store original label mapping
        
    def train(self, X, y):
        """Train multiple models and combine - FIXED label handling"""
        
        # Save feature names
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_array)
        
        # FIXED: Store the original labels
        unique_labels = np.unique(y)
        print(f"🏷️ Unique labels in data: {unique_labels}")
        
        # Create label mapping: model output -> original label
        self.label_mapping = {i: label for i, label in enumerate(sorted(unique_labels))}
        reverse_mapping = {label: i for i, label in self.label_mapping.items()}
        
        print(f"🔄 Label mapping: {self.label_mapping}")
        
        # Map to consecutive integers
        if hasattr(y, 'map'):
            y_mapped = y.map(reverse_mapping).astype(int)
        else:
            y_mapped = np.array([reverse_mapping[val] for val in y])
        
        print(f"✅ Mapped labels: {np.unique(y_mapped)}")
        
        print("🎓 Training ensemble models...")
        
        # XGBoost with proper multi-class configuration
        num_classes = len(unique_labels)
        
        if num_classes == 1:
            print("  ⚠️ Single class detected, using dummy binary classification")
            objective = 'binary:logistic'
            xgb_params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': objective,
                'random_state': 42,
                'verbosity': 0
            }
        elif num_classes == 2:
            objective = 'binary:logistic'
            xgb_params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': objective,
                'random_state': 42,
                'verbosity': 0
            }
        else:
            objective = 'multi:softprob'
            xgb_params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': objective,
                'num_class': num_classes,
                'random_state': 42,
                'verbosity': 0
            }
            
        self.models['xgb'] = xgb.XGBClassifier(**xgb_params)
        
        # Prepare training data
        if num_classes == 1:
            X_train = np.vstack([X_scaled, X_scaled[:3]])
            y_train = np.hstack([y_mapped, [1-y_mapped[0]] * 3])
            print("  ⚠️ Single class detected, using dummy samples")
        else:
            X_train = X_scaled
            y_train = y_mapped
        
        # Train models
        self.models['xgb'].fit(X_train, y_train)
        print("  ✅ XGBoost trained")
        
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        self.models['rf'].fit(X_train, y_train)
        print("  ✅ Random Forest trained")
        
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        self.models['gb'].fit(X_train, y_train)
        print("  ✅ Gradient Boosting trained")
        
        # Evaluate
        y_pred = self.predict(X)
        accuracy = accuracy_score(y_mapped, y_pred)
        print(f"  📊 Ensemble accuracy: {accuracy*100:.2f}%")
        
        return self
    
    def predict(self, X):
        """Predict using ensemble voting - returns original labels"""
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
            
        X_scaled = self.scaler.transform(X_array)
        
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Majority voting
        predictions = np.array(predictions)
        final_pred = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=0,
            arr=predictions
        )
        
        return final_pred
    
    def predict_proba(self, X):
        """Get probability estimates"""
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
            
        X_scaled = self.scaler.transform(X_array)
        
        probas = []
        for name, model in self.models.items():
            proba = model.predict_proba(X_scaled)
            probas.append(proba)
        
        avg_proba = np.mean(probas, axis=0)
        return avg_proba
    
    def get_original_label(self, model_output):
        """Convert model output back to original label"""
        return self.label_mapping.get(int(model_output), 0)

# ===============================
# 💰 KELLY CRITERION POSITION SIZING
# ===============================
class KellySizer:
    @staticmethod
    def calculate(win_rate, avg_win, avg_loss, confidence, max_fraction=0.25):
        """Kelly Criterion for optimal position sizing"""
        
        if win_rate <= 0 or win_rate >= 1 or avg_loss <= 0:
            return 0.1
        
        adjusted_win_rate = win_rate * confidence
        win_loss_ratio = avg_win / avg_loss
        kelly = (adjusted_win_rate * win_loss_ratio - (1 - adjusted_win_rate)) / win_loss_ratio
        fractional_kelly = kelly * 0.25
        position_fraction = np.clip(fractional_kelly, 0.05, max_fraction)
        
        return position_fraction

# ===============================
# 📈 STATE MANAGEMENT
# ===============================
class TradingState:
    def __init__(self):
        self.state_file = TradingConfig.STATE_FILE
        self.load_state()
    
    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.daily_trades = data.get('daily_trades', 0)
                    self.daily_start_capital = data.get('daily_start_capital', TradingConfig.INITIAL_CAPITAL)
                    self.last_trade_date = data.get('last_trade_date', datetime.now().strftime('%Y-%m-%d'))
                    self.total_trades = data.get('total_trades', 0)
                    self.winning_trades = data.get('winning_trades', 0)
                    self.losing_trades = data.get('losing_trades', 0)
                    self.entry_price = data.get('entry_price', None)
                    self.total_pnl = data.get('total_pnl', 0.0)
                    self.win_amounts = data.get('win_amounts', [])
                    self.loss_amounts = data.get('loss_amounts', [])
                    print(f"✅ Loaded state: {self.total_trades} trades, ROI: {self.total_pnl:.2f}")
            except Exception as e:
                print(f"⚠️ Could not load state: {e}")
                self.reset_state()
        else:
            self.reset_state()
    
    def reset_state(self):
        self.reset_daily()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.entry_price = None
        self.total_pnl = 0.0
        self.win_amounts = []
        self.loss_amounts = []
    
    def save_state(self):
        try:
            data = {
                'daily_trades': self.daily_trades,
                'daily_start_capital': self.daily_start_capital,
                'last_trade_date': self.last_trade_date,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'entry_price': self.entry_price,
                'total_pnl': self.total_pnl,
                'win_amounts': self.win_amounts,
                'loss_amounts': self.loss_amounts
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save state: {e}")
    
    def reset_daily(self):
        self.daily_trades = 0
        self.daily_start_capital = TradingConfig.INITIAL_CAPITAL
        self.last_trade_date = datetime.now().strftime('%Y-%m-%d')
    
    def check_new_day(self):
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self.last_trade_date:
            self.reset_daily()
            self.save_state()
    
    def get_win_rate(self):
        if self.total_trades == 0:
            return 0.5
        return self.winning_trades / self.total_trades
    
    def get_avg_win(self):
        if not self.win_amounts:
            return 0.05
        return np.mean(self.win_amounts)
    
    def get_avg_loss(self):
        if not self.loss_amounts:
            return 0.03
        return abs(np.mean(self.loss_amounts))

# ===============================
# 🛡️ SAFETY MANAGER
# ===============================
class SafetyManager:
    @staticmethod
    def can_trade(state, current_capital):
        daily_loss = state.daily_start_capital - current_capital
        if daily_loss > TradingConfig.MAX_DAILY_LOSS:
            print(f"🚨 Daily loss limit: ${daily_loss:.2f}")
            return False, "daily_loss"
        
        total_loss_pct = ((TradingConfig.INITIAL_CAPITAL - current_capital) / 
                         TradingConfig.INITIAL_CAPITAL) * 100
        if total_loss_pct > TradingConfig.MAX_TOTAL_LOSS_PCT:
            print(f"🚨 Total loss limit: {total_loss_pct:.1f}%")
            return False, "total_loss"
        
        if current_capital < TradingConfig.MIN_CAPITAL:
            print(f"🚨 Capital too low: ${current_capital:.2f}")
            return False, "min_capital"
        
        if state.daily_trades >= TradingConfig.MAX_TRADES_PER_DAY:
            return False, "trade_limit"
        
        return True, "ok"
    
    @staticmethod
    def check_stop_loss(current_price, entry_price, volatility_mult=1.0):
        if not entry_price or entry_price <= 0:
            return False
        
        adjusted_stop = TradingConfig.BASE_STOP_LOSS_PCT * volatility_mult
        price_change_pct = ((current_price - entry_price) / entry_price) * 100
        
        if price_change_pct <= -adjusted_stop:
            print(f"🛑 Stop loss: {price_change_pct:.2f}%")
            return True
        return False
    
    @staticmethod
    def check_take_profit(current_price, entry_price, volatility_mult=1.0):
        if not entry_price or entry_price <= 0:
            return False
        
        adjusted_tp = TradingConfig.BASE_TAKE_PROFIT_PCT * volatility_mult
        price_change_pct = ((current_price - entry_price) / entry_price) * 100
        
        if price_change_pct >= adjusted_tp:
            print(f"🎯 Take profit: {price_change_pct:.2f}%")
            return True
        return False

# ===============================
# 🔌 EXCHANGE & BALANCE
# ===============================
def initialize_exchange():
    API_KEY = "lff0bNg0UoHnmu72Gdc5qF572hHWRRmakMA2wGqtUxMfaMpCTWxuIv5j4ZF1MnIn"
    API_SECRET = "XjOrXtG2fgMsS0brfm6gxPMIsd6vKuih7qEmCqi2Jc0MqPP7WiC6tk58Nvh00uC8"
    
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True
    })
    
    if TradingConfig.LIVE_TRADING:
        print("🔴 LIVE TRADING MODE")
        try:
            balance = exchange.fetch_balance()
            usdt_bal = balance.get('USDT', {}).get('free', 0) if isinstance(balance.get('USDT'), dict) else 0
            btc_bal = balance.get('BTC', {}).get('free', 0) if isinstance(balance.get('BTC'), dict) else 0
            print(f"✅ Connected | USDT: ${usdt_bal:.2f} | BTC: {btc_bal:.6f}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect: {e}")
    else:
        try:
            exchange.set_sandbox_mode(True)
            print("✅ TESTNET MODE (Paper Trading)")
        except:
            print("✅ SIMULATION MODE")
    
    return exchange

class BalanceManager:
    def __init__(self, live=False):
        self.live = live
        self.virtual_btc = 0.1  # Start with no BTC
        self.virtual_usdt = TradingConfig.INITIAL_CAPITAL
    
    def get_balance(self, exchange):
        if self.live:
            try:
                bal = exchange.fetch_balance()
                btc = bal.get('BTC', {}).get('free', 0) if isinstance(bal.get('BTC'), dict) else 0
                usdt = bal.get('USDT', {}).get('free', 0) if isinstance(bal.get('USDT'), dict) else 0
                return {'btc': btc, 'usdt': usdt}
            except:
                return {'btc': 0, 'usdt': 0}
        return {'btc': self.virtual_btc, 'usdt': self.virtual_usdt}
    
    def update(self, btc_change, usdt_change):
        self.virtual_btc += btc_change
        self.virtual_usdt += usdt_change

# ===============================
# 📊 DATA & FEATURES
# ===============================
def fetch_data(exchange, symbol="BTC/USDT", timeframe="1h", limit=500):
    for attempt in range(3):
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if len(bars) < 100:
                raise ValueError(f"Only {len(bars)} candles")
            
            df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            print(f"✅ Fetched {len(df)} candles")
            return df
        except Exception as e:
            print(f"⚠️ Fetch attempt {attempt+1}: {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                raise

def prepare_features(df):
    """Calculate all features"""
    df = AdvancedIndicators.calculate_all(df)
    df = AdvancedIndicators.add_regime_features(df)
    df = SmartLabeling.adaptive_threshold(df)
    
    # Drop rows with NaN
    df.dropna(subset=['close', 'signal'], inplace=True)
    
    return df

def get_feature_columns():
    """Define feature set"""
    features = [
        # Moving averages
        'SMA10', 'SMA20', 'SMA50', 'SMA100', 'SMA200',
        'EMA10', 'EMA20', 'EMA50', 'EMA100',
        
        # MACD
        'MACD', 'MACD_signal', 'MACD_hist',
        
        # Bollinger
        'BB_width', 'BB_position',
        
        # Oscillators
        'RSI', 'RSI_oversold', 'RSI_overbought',
        'Stoch_K', 'Stoch_D',
        'CCI', 'Williams_R',
        
        # Volatility
        'ATR', 'ATR_pct', 'Volatility', 'Volatility_ratio',
        
        # Momentum
        'ROC', 'Momentum',
        
        # Volume
        'Volume_ratio', 'OBV', 'OBV_EMA',
        
        # Price patterns
        'Body_size', 'Upper_shadow', 'Lower_shadow',
        'Higher_high', 'Lower_low',
        
        # Trend
        'ADX', 'Plus_DI', 'Minus_DI',
        
        # Returns
        'Return_1', 'Return_5', 'Return_10',
        
        # Distance from MAs
        'Distance_SMA20', 'Distance_SMA50', 'Distance_SMA200',
        
        # Crossovers
        'SMA_cross_20_50', 'EMA_cross_10_50',
        
        # Regime
        'Trending', 'Strong_trend',
        'High_volatility', 'Low_volatility',
        'Bull_market', 'Bear_market',
        'Range_position'
    ]
    
    return features

# ===============================
# 🎓 MODEL TRAINING
# ===============================
def train_model(df):
    """Train ensemble model with proper validation"""
    
    features = get_feature_columns()
    
    # Ensure all features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"⚠️ Missing features: {missing_features[:5]}...")
        for f in missing_features:
            df[f] = 0
    
    X = df[features].fillna(0)
    y = df['signal']
    
    if len(X) < 50:
        raise ValueError(f"Not enough data to train: only {len(X)} samples (need 50+)")
    
    # Balance dataset
    unique, counts = np.unique(y, return_counts=True)
    print(f"📊 Class distribution: {dict(zip(unique, counts))}")
    
    if len(unique) > 1 and counts.min() >= 2:
        try:
            k_neighbors = min(5, counts.min()-1)
            if k_neighbors >= 1:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X, y = smote.fit_resample(X, y)
                print(f"✅ SMOTE applied: {len(X)} samples")
        except Exception as e:
            print(f"⚠️ SMOTE skipped: {e}")
    
    # Train ensemble
    model = EnsembleModel()
    model.train(X, y)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(model, f"model_{timestamp}.pkl")
    print(f"💾 Model saved: model_{timestamp}.pkl")
    
    return model

# ===============================
# 💱 TRADE EXECUTION - FIXED
# ===============================
def execute_trade(exchange, balance_mgr, state, decision, confidence, current_price, volatility_mult):
    """Execute trade with Kelly sizing - FIXED logic"""
    
    balance = balance_mgr.get_balance(exchange)
    btc = balance['btc']
    usdt = balance['usdt']
    capital = usdt + btc * current_price
    
    print(f"📊 Balance: {btc:.6f} BTC | ${usdt:.2f} USDT | Total: ${capital:.2f}")
    
    # Check stop loss / take profit FIRST (highest priority)
    if state.entry_price and btc > 0:
        if SafetyManager.check_stop_loss(current_price, state.entry_price, volatility_mult):
            decision = "sell"
            confidence = 1.0
            print("🛑 STOP LOSS TRIGGERED - Overriding to SELL")
        elif SafetyManager.check_take_profit(current_price, state.entry_price, volatility_mult):
            decision = "sell"
            confidence = 1.0
            print("🎯 TAKE PROFIT TRIGGERED - Overriding to SELL")
    
    # Confidence filter (but NOT for stop loss/take profit sells)
    if confidence < TradingConfig.MIN_CONFIDENCE:
        if decision == "sell" and state.entry_price:
            # Allow low-confidence sells if we have a position
            print(f"⚠️ Low confidence ({confidence:.3f}), but allowing sell to close position")
        else:
            print(f"⚠️ Low confidence ({confidence:.3f}), skipping trade")
            return False
    
    action = "HOLD"
    trade_executed = False
    
    # BUY LOGIC - FIXED
    if decision == "buy" and usdt > TradingConfig.MIN_ORDER_SIZE:
        # Kelly position sizing
        if TradingConfig.USE_KELLY_SIZING:
            win_rate = state.get_win_rate()
            avg_win = state.get_avg_win()
            avg_loss = state.get_avg_loss()
            position_fraction = KellySizer.calculate(win_rate, avg_win, avg_loss, confidence)
        else:
            position_fraction = TradingConfig.MAX_POSITION_PCT * confidence
        
        max_buy = min(
            usdt * position_fraction,
            usdt * 0.95
        )
        
        if max_buy < TradingConfig.MIN_ORDER_SIZE:
            print(f"⏸ Order too small: ${max_buy:.2f}")
            return False
        
        btc_amount = max_buy / current_price
        
        if TradingConfig.LIVE_TRADING:
            try:
                print(f"🔴 LIVE BUY: {btc_amount:.6f} BTC (${max_buy:.2f})")
                order = exchange.create_market_buy_order('BTC/USDT', btc_amount)
                print(f"✅ Order filled: {order['id']}")
                action = "BUY"
                trade_executed = True
                state.entry_price = current_price
            except Exception as e:
                print(f"❌ Buy failed: {e}")
                return False
        else:
            balance_mgr.update(btc_amount, -max_buy)
            print(f"✅ PAPER BUY: {btc_amount:.6f} BTC @ ${current_price:.2f}")
            print(f"   💵 Spent: ${max_buy:.2f} | Position size: {position_fraction*100:.1f}%")
            action = "BUY"
            trade_executed = True
            state.entry_price = current_price
    
    # SELL LOGIC - FIXED
    elif decision == "sell" and btc > 0:
        proceeds = btc * current_price
        
        if TradingConfig.LIVE_TRADING:
            try:
                print(f"🔴 LIVE SELL: {btc:.6f} BTC (${proceeds:.2f})")
                order = exchange.create_market_sell_order('BTC/USDT', btc)
                print(f"✅ Order filled: {order['id']}")
                action = "SELL"
                trade_executed = True
            except Exception as e:
                print(f"❌ Sell failed: {e}")
                return False
        else:
            balance_mgr.update(-btc, proceeds)
            print(f"✅ PAPER SELL: {btc:.6f} BTC @ ${current_price:.2f}")
            print(f"   💵 Received: ${proceeds:.2f}")
            action = "SELL"
            trade_executed = True
        
        # Calculate P&L
        if state.entry_price and state.entry_price > 0:
            pnl = (current_price - state.entry_price) * btc
            pnl_pct = ((current_price / state.entry_price) - 1) * 100
            print(f"💰 P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            
            state.total_pnl += pnl
            
            if pnl > 0:
                state.winning_trades += 1
                state.win_amounts.append(pnl_pct / 100)
            else:
                state.losing_trades += 1
                state.loss_amounts.append(pnl_pct / 100)
        
        state.entry_price = None
    
    elif decision == "sell" and btc == 0:
        print(f"⏸ SELL signal but no BTC to sell")
    elif decision == "buy" and usdt < TradingConfig.MIN_ORDER_SIZE:
        print(f"⏸ BUY signal but insufficient USDT (${usdt:.2f})")
    else:
        print(f"⏸ HOLD - No action needed")
    
    # Update state
    if trade_executed:
        state.daily_trades += 1
        state.total_trades += 1
        state.save_state()
        
        # Log trade
        log_exists = os.path.exists(TradingConfig.LOG_FILE)
        with open(TradingConfig.LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not log_exists:
                writer.writerow(["timestamp","action","price","confidence","btc","usdt","capital","pnl"])
            
            new_balance = balance_mgr.get_balance(exchange)
            new_capital = new_balance['usdt'] + new_balance['btc'] * current_price
            
            writer.writerow([
                datetime.now().isoformat(),
                action,
                round(current_price, 2),
                round(confidence, 3),
                round(new_balance['btc'], 8),
                round(new_balance['usdt'], 2),
                round(new_capital, 2),
                round(state.total_pnl, 2)
            ])
    
    return trade_executed

# ===============================
# 📊 PERFORMANCE METRICS
# ===============================
def save_metrics(state, capital):
    """Save performance metrics"""
    
    try:
        win_rate = state.get_win_rate()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_trades': state.total_trades,
            'winning_trades': state.winning_trades,
            'losing_trades': state.losing_trades,
            'win_rate': win_rate,
            'total_pnl': state.total_pnl,
            'current_capital': capital,
            'roi': ((capital / TradingConfig.INITIAL_CAPITAL) - 1) * 100,
            'avg_win': state.get_avg_win() * 100,
            'avg_loss': state.get_avg_loss() * 100
        }
        
        with open(TradingConfig.METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    except Exception as e:
        print(f"⚠️ Could not save metrics: {e}")
        return None

# ===============================
# 🎯 MAIN LOOP - FIXED
# ===============================
def main():
    print("="*70)
    print("🚀 ULTIMATE TRADING BOT v3.1 - FIXED VERSION")
    print("="*70)
    print(f"Mode: {'🔴 LIVE' if TradingConfig.LIVE_TRADING else '📄 PAPER'} TRADING")
    print(f"Capital: ${TradingConfig.INITIAL_CAPITAL:.2f}")
    print(f"Features: {len(get_feature_columns())} indicators")
    print(f"Min Confidence: {TradingConfig.MIN_CONFIDENCE*100:.0f}%")
    print(f"Ensemble: {'✅ Enabled' if TradingConfig.USE_ENSEMBLE else '❌ Disabled'}")
    print(f"Kelly Sizing: {'✅ Enabled' if TradingConfig.USE_KELLY_SIZING else '❌ Disabled'}")
    print("="*70)
    
    if TradingConfig.LIVE_TRADING:
        print("\n⚠️  LIVE TRADING - Type 'START LIVE TRADING' in 10 seconds:")
        try:
            import select
            import sys
            i, o, e = select.select([sys.stdin], [], [], 10)
            if i and sys.stdin.readline().strip() == 'START LIVE TRADING':
                print("✅ Confirmed")
            else:
                print("❌ Not confirmed. Exiting.")
                return
        except:
            import msvcrt
            import time as time_module
            print("Press ENTER within 10 seconds to confirm...")
            start_time = time_module.time()
            confirmed = False
            while time_module.time() - start_time < 10:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\r':
                        confirmed = True
                        break
                time_module.sleep(0.1)
            
            if not confirmed:
                print("❌ Not confirmed. Exiting.")
                return
            print("✅ Confirmed")
    
    # Initialize
    exchange = initialize_exchange()
    balance_mgr = BalanceManager(TradingConfig.LIVE_TRADING)
    state = TradingState()
    
    # Initial training
    print("\n🎓 Training model...")
    df = fetch_data(exchange)
    df = prepare_features(df)
    model = train_model(df)
    last_retrain = time.time()
    print("✅ Training complete\n")
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            state.check_new_day()
            
            print(f"\n{'='*70}")
            print(f"Iteration {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")
            
            # Fetch fresh data
            df = fetch_data(exchange)
            current_price = float(df['close'].iloc[-1])
            
            # Get balance and capital
            balance = balance_mgr.get_balance(exchange)
            capital = balance['usdt'] + balance['btc'] * current_price
            
            # Safety checks
            can_trade, reason = SafetyManager.can_trade(state, capital)
            if not can_trade:
                print(f"🚫 Trading disabled: {reason}")
                time.sleep(TradingConfig.CHECK_INTERVAL)
                continue
            
            # Retrain if needed
            if time.time() - last_retrain > TradingConfig.RETRAIN_INTERVAL:
                print("🔄 Retraining model...")
                df = fetch_data(exchange)
                df = prepare_features(df)
                model = train_model(df)
                last_retrain = time.time()
            
            # Prepare data
            df = prepare_features(df)
            
            if len(df) == 0:
                print("⚠️ No data, waiting...")
                time.sleep(60)
                continue
            
            # Get features
            features = get_feature_columns()
            latest = df[features].iloc[-1:].fillna(0)
            
            # Predict - FIXED
            proba = model.predict_proba(latest)[0]
            pred_class = int(model.predict(latest)[0])
            confidence = float(proba[pred_class])
            
            # Convert model output to original label
            original_signal = model.get_original_label(pred_class)
            
            # Map to decision
            decision_map = {-1: "sell", 0: "hold", 1: "buy"}
            decision = decision_map.get(original_signal, "hold")
            
            print(f"🤖 Raw prediction: class={pred_class}, signal={original_signal}, decision={decision}")
            print(f"📊 Probabilities: {proba}")
            print(f"✨ Confidence: {confidence:.3f} (min required: {TradingConfig.MIN_CONFIDENCE:.3f})")
            
            # REMOVED the confusing smart override logic that was causing issues
            # Now it's straightforward: if model says buy and we have cash, buy
            # If model says sell and we have BTC, sell
            
            # Volatility multiplier for dynamic stops
            volatility_mult = df['Volatility_ratio'].iloc[-1] if 'Volatility_ratio' in df.columns else 1.0
            volatility_mult = np.clip(volatility_mult, 0.5, 2.0)
            
            print(f"💵 BTC Price: ${current_price:.2f}")
            print(f"📊 Volatility Mult: {volatility_mult:.2f}x")
            print(f"🎯 DECISION: {decision.upper()}")
            
            # Execute
            execute_trade(exchange, balance_mgr, state, decision, confidence, current_price, volatility_mult)
            
            # Stats
            win_rate = state.get_win_rate() * 100
            roi = ((capital / TradingConfig.INITIAL_CAPITAL) - 1) * 100
            print(f"📈 Stats: {state.total_trades} trades | {win_rate:.1f}% win rate | ROI: {roi:+.2f}%")
            
            # Save metrics
            save_metrics(state, capital)
            
            # Wait
            print(f"\n💤 Next check in {TradingConfig.CHECK_INTERVAL//60} minutes...")
            time.sleep(TradingConfig.CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
            
            print(f"\n{'='*70}")
            print("📊 FINAL RESULTS")
            print(f"{'='*70}")
            
            try:
                balance = balance_mgr.get_balance(exchange)
                final_capital = balance['usdt'] + balance['btc'] * current_price
                roi = ((final_capital / TradingConfig.INITIAL_CAPITAL) - 1) * 100
            except:
                final_capital = TradingConfig.INITIAL_CAPITAL
                roi = 0.0
            
            print(f"Initial Capital:    ${TradingConfig.INITIAL_CAPITAL:.2f}")
            print(f"Final Capital:      ${final_capital:.2f}")
            print(f"Total P&L:          ${state.total_pnl:+.2f}")
            print(f"ROI:                {roi:+.2f}%")
            print(f"Total Trades:       {state.total_trades}")
            print(f"Winning Trades:     {state.winning_trades}")
            print(f"Losing Trades:      {state.losing_trades}")
            
            win_rate = state.get_win_rate() * 100
            print(f"Win Rate:           {win_rate:.1f}%")
            print(f"Avg Win:            {state.get_avg_win()*100:+.2f}%")
            print(f"Avg Loss:           {state.get_avg_loss()*100:.2f}%")
            print(f"{'='*70}")
            break
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔚 Bot stopped")
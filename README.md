# 🤖 Ultimate Trading Bot - Complete Setup

## 🚀 Quick Start (2 Minutes)

### Option 1: Automated Setup
```bash
# Make script executable
chmod +x quick_start.sh

# Run setup
./quick_start.sh

# That's it! Bot is running.
```

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip3 install ccxt pandas numpy scikit-learn xgboost imbalanced-learn matplotlib joblib

# 2. Run bot (paper trading by default)
python3 ultimate_bot.py &

# 3. Check status
python3 monitor.py
```

---

## 📊 Daily Monitoring (30 seconds)

### Check Dashboard
```bash
python3 monitor.py
```

**You'll see:**
```
🤖 TRADING BOT MONITORING DASHBOARD
====================================
🔧 Bot Status: ✅ RUNNING

📊 OVERVIEW
------------------------------
💰 Current Capital:     $112.45
📈 Total ROI:           +12.45%
🎯 Win Rate:            56.7%
🔢 Total Trades:        47
   ├─ Wins:             28
   └─ Losses:           19

🏥 HEALTH CHECK
------------------------------
✅ Win rate healthy (56.7%)
✅ Profitable (+12.45%)
✅ Sufficient trade history (47 trades)

💡 RECOMMENDATIONS:
  → Bot performing well, continue monitoring weekly
```

### Quick Health Check
```bash
# Is bot running?
ps aux | grep ultimate_bot.py

# Latest metrics
cat performance_metrics.json

# Last 5 trades
tail -5 trades_log.csv
```

---

## 🎮 Control Commands

### Pause Trading (Temporary)
```bash
touch PAUSE_TRADING.txt
# Bot stops trading but keeps running

# Resume
rm PAUSE_TRADING.txt
```

### Stop Bot (Emergency)
```bash
touch EMERGENCY_STOP.txt
# Bot stops immediately and exits
```

### Kill Process (Force stop)
```bash
pkill -9 -f ultimate_bot.py
```

### Restart Bot
```bash
# Stop
touch EMERGENCY_STOP.txt
# Wait for it to stop
sleep 5
# Remove file
rm EMERGENCY_STOP.txt
# Restart
python3 ultimate_bot.py &
```

---

## 📁 File Structure

```
trading-bot/
├── ultimate_bot.py          # Main bot
├── monitor.py               # Dashboard
├── quick_start.sh           # Setup script
├── trading_state.json       # Bot state (auto-saved)
├── performance_metrics.json # Current stats (auto-updated)
├── trades_log.csv          # All trades (auto-logged)
├── model_*.pkl             # Trained models (auto-saved)
├── scaler_*.pkl            # Feature scalers (auto-saved)
└── logs/
    └── bot.log             # Detailed logs
```

### What Gets Auto-Created:
- ✅ `trading_state.json` - First trade
- ✅ `performance_metrics.json` - First iteration
- ✅ `trades_log.csv` - First trade
- ✅ `model_*.pkl` - First training
- ✅ `scaler_*.pkl` - First training

### What You Create Manually:
- `PAUSE_TRADING.txt` - To pause
- `EMERGENCY_STOP.txt` - To stop
- `.env` - API keys (if live trading)

---

## ⚙️ Configuration

Edit `TradingConfig` class in `ultimate_bot.py`:

```python
class TradingConfig:
    # Mode
    LIVE_TRADING = False  # ← Change to True for real money
    
    # Capital
    INITIAL_CAPITAL = 100.0  # ← Your starting amount
    
    # Risk (adjust these based on results)
    BASE_STOP_LOSS_PCT = 5.0      # Stop loss trigger
    BASE_TAKE_PROFIT_PCT = 10.0   # Take profit trigger
    MAX_POSITION_PCT = 0.45       # Max 45% per trade
    MIN_CONFIDENCE = 0.6          # Min 60% confidence
    
    # Features (usually leave as True)
    USE_ENSEMBLE = True           # Use 3 models
    USE_KELLY_SIZING = True       #
#!/usr/bin/env python3
"""
Trading Bot Monitoring Dashboard
Run this anytime to check your bot's performance
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import subprocess
import psutil

# ===============================
# 🎨 COLORS
# ===============================
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def green(text):
    return f"{Colors.GREEN}{text}{Colors.END}"

def red(text):
    return f"{Colors.RED}{text}{Colors.END}"

def yellow(text):
    return f"{Colors.YELLOW}{text}{Colors.END}"

def blue(text):
    return f"{Colors.BLUE}{text}{Colors.END}"

def bold(text):
    return f"{Colors.BOLD}{text}{Colors.END}"

def cyan(text):
    return f"{Colors.CYAN}{text}{Colors.END}"

# ===============================
# 📊 LOAD DATA
# ===============================
def load_metrics():
    """Load performance metrics"""
    if not os.path.exists("performance_metrics.json"):
        return None
    
    with open("performance_metrics.json", 'r') as f:
        return json.load(f)

def load_state():
    """Load trading state"""
    if not os.path.exists("trading_state.json"):
        return None
    
    with open("trading_state.json", 'r') as f:
        return json.load(f)

def load_trades():
    """Load trade history"""
    if not os.path.exists("trades_log.csv"):
        return None
    
    try:
        df = pd.read_csv("trades_log.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except:
        return None

# ===============================
# 📈 ANALYTICS
# ===============================
def calculate_sharpe_ratio(trades_df):
    """Calculate Sharpe ratio"""
    if trades_df is None or len(trades_df) < 2:
        return 0
    
    # Get returns from trades
    trades_df = trades_df.copy()
    trades_df['returns'] = trades_df['capital'].pct_change()
    trades_df = trades_df.dropna()
    
    if len(trades_df) == 0:
        return 0
    
    avg_return = trades_df['returns'].mean()
    std_return = trades_df['returns'].std()
    
    if std_return == 0:
        return 0
    
    # Annualized Sharpe (assuming ~250 trading days)
    sharpe = (avg_return / std_return) * np.sqrt(250)
    return sharpe

def calculate_max_drawdown(trades_df):
    """Calculate maximum drawdown"""
    if trades_df is None or len(trades_df) < 2:
        return 0
    
    trades_df = trades_df.copy()
    trades_df['cummax'] = trades_df['capital'].cummax()
    trades_df['drawdown'] = (trades_df['capital'] - trades_df['cummax']) / trades_df['cummax'] * 100
    
    return trades_df['drawdown'].min()

def get_recent_performance(trades_df, days=7):
    """Get performance for last N days"""
    if trades_df is None or len(trades_df) < 2:
        return None
    
    cutoff = datetime.now() - timedelta(days=days)
    recent = trades_df[trades_df['timestamp'] >= cutoff]
    
    if len(recent) < 2:
        return None
    
    start_capital = recent.iloc[0]['capital']
    end_capital = recent.iloc[-1]['capital']
    roi = ((end_capital - start_capital) / start_capital) * 100
    
    # Count wins/losses
    recent_trades = recent[recent['action'].isin(['BUY', 'SELL'])]
    wins = (recent['pnl'] > 0).sum()
    losses = (recent['pnl'] < 0).sum()
    
    return {
        'trades': len(recent_trades),
        'roi': roi,
        'wins': wins,
        'losses': losses,
        'win_rate': wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    }

def get_hourly_stats(trades_df):
    """Analyze best trading hours"""
    if trades_df is None or len(trades_df) < 10:
        return None
    
    try:
        trades_df = trades_df.copy()
        trades_df['hour'] = trades_df['timestamp'].dt.hour
        
        hourly = trades_df.groupby('hour').agg({
            'pnl': ['count', 'mean']
        }).round(2)
        
        return hourly
    except:
        return None

def bot_status():
    """Check if ultimate_bot.py is running"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'])
                if 'ultimate_bot.py' in cmdline:
                    return True
        return False
    except Exception as e:
        print(f"Error checking bot status: {e}")
        return False

# ===============================
# 🎨 DISPLAY
# ===============================
def print_header():
    """Print dashboard header"""
    print("\n" + "="*70)
    print(bold(cyan("🤖 TRADING BOT MONITORING DASHBOARD")))
    print("="*70)
    print(f"📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

def print_bot_status():
    """Print bot status"""
    running = bot_status()
    
    if running:
        status = green("✅ RUNNING")
    else:
        status = red("❌ STOPPED")
    
    print(bold(f"🔧 Bot Status: {status}"))
    
    # Check for control files
    if os.path.exists("EMERGENCY_STOP.txt"):
        print(red("   ⚠️  EMERGENCY_STOP.txt detected - bot will stop"))
    if os.path.exists("PAUSE_TRADING.txt"):
        print(yellow("   ⏸️  PAUSE_TRADING.txt detected - trading paused"))
    
    print()

def print_overview(metrics, state):
    """Print overview section"""
    print(bold("📊 OVERVIEW"))
    print("-" * 70)
    
    if metrics:
        capital = metrics['current_capital']
        roi = metrics['roi']
        win_rate = metrics['win_rate'] * 100
        
        # Color based on performance
        capital_color = green if capital >= 100 else red
        roi_color = green if roi >= 0 else red
        wr_color = green if win_rate >= 50 else (yellow if win_rate >= 45 else red)
        
        print(f"💰 Current Capital:     {capital_color(f'${capital:.2f}')}")
        print(f"📈 Total ROI:           {roi_color(f'{roi:+.2f}%')}")
        print(f"🎯 Win Rate:            {wr_color(f'{win_rate:.1f}%')}")
        print(f"🔢 Total Trades:        {metrics['total_trades']}")
        print(f"   ├─ Wins:             {green(str(metrics['winning_trades']))}")
        print(f"   └─ Losses:           {red(str(metrics['losing_trades']))}")
        
        if metrics['total_pnl'] != 0:
            pnl_color = green if metrics['total_pnl'] > 0 else red
            print(f"💵 Total P&L:           {pnl_color(f'${metrics['total_pnl']:+.2f}')}")
    else:
        print(red("No metrics available - bot hasn't traded yet"))
    
    print()

def print_performance_metrics(metrics, trades_df):
    """Print detailed performance metrics"""
    print(bold("📈 PERFORMANCE METRICS"))
    print("-" * 70)
    
    if metrics:
        # Average win/loss
        avg_win = metrics.get('avg_win', 0)
        avg_loss = metrics.get('avg_loss', 0)
        
        print(f"📊 Average Win:         {green(f'+{avg_win:.2f}%')}")
        print(f"📊 Average Loss:        {red(f'{avg_loss:.2f}%')}")
        
        # Win/Loss ratio
        if avg_loss != 0:
            wl_ratio = abs(avg_win / avg_loss)
            ratio_color = green if wl_ratio > 1 else red
            print(f"⚖️  Win/Loss Ratio:     {ratio_color(f'{wl_ratio:.2f}')}")
        
        # Expectancy
        win_rate = metrics['win_rate']
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        exp_color = green if expectancy > 0 else red
        print(f"🎲 Expectancy:          {exp_color(f'{expectancy:+.2f}%')}")
        
        # Sharpe ratio
        sharpe = calculate_sharpe_ratio(trades_df)
        sharpe_color = green if sharpe > 1 else (yellow if sharpe > 0 else red)
        print(f"📉 Sharpe Ratio:        {sharpe_color(f'{sharpe:.2f}')}")
        
        # Max drawdown
        max_dd = calculate_max_drawdown(trades_df)
        dd_color = green if max_dd > -10 else (yellow if max_dd > -20 else red)
        print(f"📉 Max Drawdown:        {dd_color(f'{max_dd:.2f}%')}")
    else:
        print(red("No performance data available yet"))
    
    print()

def print_recent_performance(trades_df):
    """Print recent performance"""
    print(bold("⏰ RECENT PERFORMANCE"))
    print("-" * 70)
    
    for days in [1, 7, 30]:
        perf = get_recent_performance(trades_df, days)
        
        if perf:
            roi_color = green if perf['roi'] >= 0 else red
            wr_color = green if perf['win_rate'] >= 50 else red
            
            print(f"Last {days:2d} days: {perf['trades']:3d} trades | "
                  f"ROI: {roi_color(f'{perf['roi']:+.2f}%'):20s} | "
                  f"Win Rate: {wr_color(f'{perf['win_rate']:.1f}%')}")
        else:
            print(f"Last {days:2d} days: {yellow('No data')}")
    
    print()

def print_recent_trades(trades_df, n=10):
    """Print last N trades"""
    print(bold(f"📝 LAST {n} TRADES"))
    print("-" * 70)
    
    if trades_df is None or len(trades_df) == 0:
        print(yellow("No trades yet"))
        print()
        return
    
    recent = trades_df.tail(n)
    
    for _, trade in recent.iterrows():
        timestamp = trade['timestamp'].strftime('%Y-%m-%d %H:%M')
        action = trade['action']
        price = trade['price']
        confidence = trade['confidence']
        pnl = trade.get('pnl', 0)
        
        # Color coding
        if action == "BUY":
            action_str = green(f"{action:4s}")
        elif action == "SELL":
            action_str = red(f"{action:4s}")
        else:
            action_str = yellow(f"{action:4s}")
        
        pnl_str = ""
        if pnl != 0:
            pnl_color = green if pnl > 0 else red
            pnl_str = f" | P&L: {pnl_color(f'${pnl:+.2f}')}"
        
        print(f"{timestamp} | {action_str} @ ${price:,.2f} | "
              f"Conf: {confidence:.2f}{pnl_str}")
    
    print()

def print_health_check(metrics, state, trades_df):
    """Print health check and recommendations"""
    print(bold("🏥 HEALTH CHECK & RECOMMENDATIONS"))
    print("-" * 70)
    
    issues = []
    warnings = []
    ok = []
    
    if metrics:
        # Check win rate
        win_rate = metrics['win_rate'] * 100
        if win_rate < 40:
            issues.append(f"❌ Win rate very low ({win_rate:.1f}%) - Consider stopping")
        elif win_rate < 50:
            warnings.append(f"⚠️  Win rate below 50% ({win_rate:.1f}%) - Monitor closely")
        else:
            ok.append(f"✅ Win rate healthy ({win_rate:.1f}%)")
        
        # Check ROI
        roi = metrics['roi']
        if roi < -20:
            issues.append(f"❌ Large loss ({roi:.1f}%) - Consider stopping")
        elif roi < -10:
            warnings.append(f"⚠️  Significant loss ({roi:.1f}%) - Review strategy")
        elif roi > 0:
            ok.append(f"✅ Profitable ({roi:+.1f}%)")
        
        # Check total trades
        total = metrics['total_trades']
        if total < 10:
            warnings.append(f"⚠️  Limited data ({total} trades) - Need more trades to evaluate")
        else:
            ok.append(f"✅ Sufficient trade history ({total} trades)")
        
        # Check recent performance
        recent = get_recent_performance(trades_df, 7)
        if recent and recent['trades'] > 5:
            if recent['win_rate'] < 40:
                issues.append(f"❌ Recent win rate very poor ({recent['win_rate']:.1f}%)")
            elif recent['win_rate'] < 50:
                warnings.append(f"⚠️  Recent win rate declining ({recent['win_rate']:.1f}%)")
        
        # Check drawdown
        max_dd = calculate_max_drawdown(trades_df)
        if max_dd < -25:
            issues.append(f"❌ Severe drawdown ({max_dd:.1f}%)")
        elif max_dd < -15:
            warnings.append(f"⚠️  Significant drawdown ({max_dd:.1f}%)")
        else:
            ok.append(f"✅ Drawdown acceptable ({max_dd:.1f}%)")
    
    # Print results
    if issues:
        print(red(bold("CRITICAL ISSUES:")))
        for issue in issues:
            print(f"  {issue}")
        print()
    
    if warnings:
        print(yellow(bold("WARNINGS:")))
        for warning in warnings:
            print(f"  {warning}")
        print()
    
    if ok:
        print(green(bold("HEALTHY INDICATORS:")))
        for item in ok:
            print(f"  {item}")
        print()
    
    # Recommendations
    print(bold("💡 RECOMMENDATIONS:"))
    
    if issues:
        print(red("  → STOP THE BOT and review settings"))
        print("     Consider: Increasing MIN_CONFIDENCE or reducing MAX_POSITION_PCT")
    elif warnings:
        print(yellow("  → Monitor closely for next few days"))
        print("     Consider: Adjusting confidence threshold or position sizing")
    else:
        print(green("  → Bot performing well, continue monitoring weekly"))
    
    print()

def print_statistics(trades_df):
    """Print additional statistics"""
    if trades_df is None or len(trades_df) < 5:
        return
    
    print(bold("📊 ADDITIONAL STATISTICS"))
    print("-" * 70)
    
    try:
        # Longest win/loss streaks
        trades_df = trades_df.copy()
        trades_df['win'] = (trades_df['pnl'] > 0).astype(int)
        
        # Calculate streaks
        trades_df['streak'] = trades_df['win'].ne(trades_df['win'].shift()).cumsum()
        win_streaks = trades_df[trades_df['win'] == 1].groupby('streak').size()
        loss_streaks = trades_df[trades_df['win'] == 0].groupby('streak').size()
        
        longest_win = win_streaks.max() if len(win_streaks) > 0 else 0
        longest_loss = loss_streaks.max() if len(loss_streaks) > 0 else 0
        
        print(f"🔥 Longest Win Streak:  {green(str(longest_win))}")
        print(f"❄️  Longest Loss Streak: {red(str(longest_loss))}")
        
        # Best and worst trades
        if len(trades_df[trades_df['pnl'] != 0]) > 0:
            best_trade = trades_df[trades_df['pnl'] != 0]['pnl'].max()
            worst_trade = trades_df[trades_df['pnl'] != 0]['pnl'].min()
            
            print(f"🏆 Best Trade:          {green(f'${best_trade:+.2f}')}")
            print(f"💀 Worst Trade:         {red(f'${worst_trade:+.2f}')}")
        
        print()
    except Exception as e:
        print(f"⚠️ Could not calculate statistics: {e}")
        print()

def print_footer():
    """Print footer with instructions"""
    print("="*70)
    print(bold("🛠️  QUICK ACTIONS"))
    print("-" * 70)
    print("📊 Refresh dashboard:         python monitor.py")
    print("⏸️  Pause trading:            touch PAUSE_TRADING.txt")
    print("▶️  Resume trading:           rm PAUSE_TRADING.txt")
    print("🛑 Emergency stop:           touch EMERGENCY_STOP.txt")
    print("📝 View full log:            tail -f bot.log")
    print("📈 View all trades:          cat trades_log.csv")
    print("="*70 + "\n")

# ===============================
# 🚀 MAIN
# ===============================
def main():
    """Main dashboard function"""
    
    # Clear screen (optional)
    # os.system('clear' if os.name == 'posix' else 'cls')
    
    # Load data
    metrics = load_metrics()
    state = load_state()
    trades_df = load_trades()
    
    # Print dashboard
    print_header()
    print_bot_status()
    print_overview(metrics, state)
    print_performance_metrics(metrics, trades_df)
    print_recent_performance(trades_df)
    print_recent_trades(trades_df, n=10)
    print_health_check(metrics, state, trades_df)
    print_statistics(trades_df)
    print_footer()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard closed\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
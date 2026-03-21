 
import os
import sys
 
# ── Config ────────────────────────────────────────────────────────────────────
 
TICKER          = "AAPL"       # try "MSFT", "RELIANCE.NS", "^NSEI", "BTC-USD"
START           = "2018-01-01"
END             = "2024-01-01"
INITIAL_CAPITAL = 100_000.0    # starting cash
COMMISSION      = 0.001        # 0.1% per trade (10 basis points)
SLIPPAGE        = 0.0005       # 0.05% per fill
OUTPUT_DIR      = "charts"     # folder where charts are saved
ML_TRAIN_RATIO  = 0.60         # 60% of data used for training the ML model
 
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    print("=" * 60)
    print("   END-TO-END QUANTITATIVE TRADING SYSTEM")
    print("=" * 60)
 
    #  Step 1: Fetch Data 
    from data_ingestion import fetch_data
    df_raw = fetch_data(TICKER, START, END)
 
    #  Step 2: Build Features 
    from feature_engineering import build_features
    df = build_features(df_raw)
 
    # Step 3: Generate Signals 
    print("\n[Strategies] Generating signals...")
    from strategies import strategy_momentum, strategy_mean_reversion, strategy_ml
 
    sig_momentum = strategy_momentum(df, allow_short=False)
    sig_meanrev  = strategy_mean_reversion(df, entry_z=1.5, exit_z=0.5, allow_short=True)
    sig_ml       = strategy_ml(df, train_ratio=ML_TRAIN_RATIO, n_estimators=200)
 
    signals = {
        "Momentum":  sig_momentum,
        "Mean Rev":  sig_meanrev,
        "ML":        sig_ml,
    }
 
    # Step 4: Backtest Each Strategy 
    print("\n[Backtester] Running backtests...")
    from backtester import run_backtest, buy_and_hold
 
    r_momentum = run_backtest(df, sig_momentum, "Momentum",
                              INITIAL_CAPITAL, COMMISSION, SLIPPAGE)
    r_meanrev  = run_backtest(df, sig_meanrev,  "Mean Rev",
                              INITIAL_CAPITAL, COMMISSION, SLIPPAGE)
    r_ml       = run_backtest(df, sig_ml,       "ML",
                              INITIAL_CAPITAL, COMMISSION, SLIPPAGE)
    r_bh       = buy_and_hold(df, INITIAL_CAPITAL, COMMISSION, SLIPPAGE)
 
    all_results = [r_momentum, r_meanrev, r_ml, r_bh]
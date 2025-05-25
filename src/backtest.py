"""
Backtesting module for evaluating trading strategy performance
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker # Added for currency formatting
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Import necessary classes and configurations
# Corrected to use relative imports for trading_strategy and feature_engineering
try:
    from .trading_strategy import TradingStrategy # Corrected to relative import
    from .feature_engineering import FeatureEngineer # Corrected to relative import
    from config import RESULTS_DIR, INTERVAL, INITIAL_CAPITAL, \
                       COMM_RATE, STOP_LOSS_PCT, SLIPPAGE, SAVE_PLOTS, \
                       MIN_DATA_DAYS, TICKER_SYMBOLS, NUM_INSTRUMENTS
except ImportError as e:
    logging.error(f"Import error: {e}. Ensure config.py, trading_strategy.py, and feature_engineering.py are correctly set up.")
    exit()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure results directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'plots').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'data').mkdir(parents=True, exist_ok=True)

class Backtester:
    def __init__(self, initial_capital=INITIAL_CAPITAL):
        """
        Initialize backtester
        
        Args:
            initial_capital (float): Initial capital for backtesting
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategy = TradingStrategy() # Initialize the strategy
        self.feature_engineer = FeatureEngineer() # To load the combined feature-engineered data
        
        # Internal state for backtest simulation
        self.open_trades = {ticker: None for ticker in TICKER_SYMBOLS} # Stores {'ticker': {'entry_price', 'entry_time', 'direction', 'units', 'stop_loss_level', 'take_profit_level'}}
        self.equity_curve_data = [] # List of {'Time': ..., 'Equity': ...} dicts
        self.trade_log_data = [] # List of trade details dicts
        self.price_data_ohlcv = None # Will store the combined OHLCV data for reference
        self.df_full_features = None # Will store the combined feature-engineered data

        # Load data during initialization
        self.load_data()
        
    def load_data(self):
        """
        Load combined feature-engineered data for backtesting.
        This data already contains all necessary OHLCV and calculated features.
        """
        logger.info("Loading combined feature-engineered data for backtesting...")
        
        combined_features_path = FEATURES_DATA_DIR / f"combined_features_{INTERVAL}_ml_strategy.csv"
        
        if not combined_features_path.exists():
            logger.error(f"Combined features file not found at {combined_features_path}. Please run feature_engineering.py first.")
            raise FileNotFoundError(f"Combined features file not found: {combined_features_path}")

        try:
            # Load with MultiIndex header
            self.df_full_features = pd.read_csv(combined_features_path, header=[0,1], index_col=0, parse_dates=True)
            logger.info(f"Loaded combined features data with shape: {self.df_full_features.shape}")
            logger.info(f"Date range: {self.df_full_features.index.min()} to {self.df_full_features.index.max()}")
        except Exception as e:
            logger.error(f"Error loading combined features data: {e}")
            raise

        # Also load the raw OHLCV data for potential reference (e.g., if we need to check original High/Low for plotting)
        # However, for strategy logic, df_full_features should contain everything.
        raw_ohlcv_path = RAW_DATA_DIR / f"combined_ohlcv_prices_{INTERVAL}.csv"
        if raw_ohlcv_path.exists():
            try:
                self.price_data_ohlcv = pd.read_csv(raw_ohlcv_path, header=[0,1], index_col=0, parse_dates=True)
                logger.info(f"Loaded raw OHLCV data with shape: {self.price_data_ohlcv.shape}")
            except Exception as e:
                logger.warning(f"Could not load raw OHLCV data: {e}. Backtest will proceed without it.")
        else:
            logger.warning(f"Raw OHLCV data not found at {raw_ohlcv_path}. Backtest will proceed without it.")
        
    def calculate_transaction_costs(self, old_units, new_units, current_price):
        """
        Calculate transaction costs for a single instrument based on unit changes.
        
        Args:
            old_units (float): Previous units held for the instrument.
            new_units (float): New units to hold for the instrument.
            current_price (float): Current price of the instrument.
            
        Returns:
            float: Transaction cost for this instrument.
        """
        # Calculate the absolute change in units
        unit_change = abs(new_units - old_units)
        
        # Value of the trade (units * price)
        trade_value = unit_change * current_price
        
        # Transaction cost
        cost = trade_value * COMM_RATE
        return cost
        
    def apply_slippage(self, units_traded, current_price):
        """
        Apply slippage cost for a single trade.
        
        Args:
            units_traded (float): Absolute number of units traded (bought or sold).
            current_price (float): Current price of the instrument.
            
        Returns:
            float: Slippage cost for this trade.
        """
        trade_value = units_traded * current_price
        slippage_cost = trade_value * SLIPPAGE
        return slippage_cost
        
    def run_backtest(self, start_date=None, end_date=None, min_history_days=MIN_DATA_DAYS):
        """
        Run comprehensive backtest simulation.
        
        Args:
            start_date (str): Start date for backtesting (e.g., "YYYY-MM-DD").
            end_date (str): End date for backtesting (e.g., "YYYY-MM-DD").
            min_history_days (int): Minimum periods of history required for strategy features.
        """
        if self.df_full_features is None:
            logger.error("Feature-engineered data not loaded. Cannot run backtest.")
            return

        # Determine the effective start and end dates for the backtest loop
        all_dates = self.df_full_features.index
        
        # Find the index where enough history is available for the first calculation
        # The earliest valid index for backtesting is where all features are non-NaN for the first time
        # This is handled by df.dropna() in feature_engineering, so we just need to ensure we have enough lookback
        # A simple approach is to start after MIN_DATA_DAYS from the beginning of the loaded data
        effective_start_idx = max(min_history_days, 1) # Ensure at least 1 for prev_data_row

        if start_date:
            start_date = pd.to_datetime(start_date)
            # Find the first index >= start_date
            idx_from_start_date = all_dates.searchsorted(start_date)
            effective_start_idx = max(effective_start_idx, idx_from_start_date)
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            # Find the last index <= end_date
            idx_to_end_date = all_dates.searchsorted(end_date, side='right') - 1
            effective_end_idx = min(len(all_dates) - 1, idx_to_end_date)
        else:
            effective_end_idx = len(all_dates) - 1

        if effective_start_idx >= effective_end_idx:
            logger.error(f"Backtest period is invalid or too short. Start index: {effective_start_idx}, End index: {effective_end_idx}. Total data points: {len(all_dates)}.")
            return

        backtest_dates = all_dates[effective_start_idx : effective_end_idx + 1]
        
        logger.info(f"Running backtest from {backtest_dates.min()} to {backtest_dates.max()}")
        logger.info(f"Number of periods to process: {len(backtest_dates)}")
        
        # Reset capital and trade logs for a fresh backtest run
        self.current_capital = self.initial_capital
        self.equity_curve_data = []
        self.trade_log_data = []
        self.open_trades = {ticker: None for ticker in TICKER_SYMBOLS}

        # The backtest loop
        for i, current_timestamp in enumerate(backtest_dates):
            # Get the actual index in the full DataFrame for current and previous data
            full_df_current_idx = self.df_full_features.index.get_loc(current_timestamp)
            full_df_prev_idx = full_df_current_idx - 1

            # Append current capital to equity curve BEFORE any trades for this period
            self.equity_curve_data.append({'Time': current_timestamp, 'Equity': self.current_capital})

            # Iterate through each ticker
            for ticker_idx, ticker in enumerate(TICKER_SYMBOLS):
                # Get the relevant data for the current ticker at this timestamp
                # Ensure the columns are correctly selected from the MultiIndex
                current_data_row = self.df_full_features.loc[current_timestamp, ticker]
                prev_data_row = self.df_full_features.loc[self.df_full_features.index[full_df_prev_idx], ticker]

                # Check for NaNs in critical columns for this specific ticker before proceeding
                # These are the columns needed for the strategy's get_my_position
                critical_cols_for_strategy = ['Close', 'Log_Close', f'Log_Close_SMA{SMA_PERIOD}', f'High_{LOOKBACK_PERIOD}_Period', f'Low_{LOOKBACK_PERIOD}_Period'] + self.strategy.ml_feature_columns
                
                # Filter current_data_row and prev_data_row to only critical columns to check for NaNs
                # This ensures we only check the columns that are actually used by the strategy
                current_critical_data = current_data_row[critical_cols_for_strategy]
                prev_critical_data = prev_data_row[critical_cols_for_strategy]

                if current_critical_data.isnull().any() or prev_critical_data.isnull().any():
                    logger.debug(f"Skipping strategy for {ticker} at {current_timestamp}: NaN values in critical data.")
                    # If there's an open trade and data is NaN, assume it closes at current price
                    if self.open_trades[ticker] is not None:
                        # Use the 'Close' price from the current_data_row for exit if available, otherwise skip
                        if 'Close' in current_data_row and not pd.isna(current_data_row['Close']):
                            self._close_trade(ticker, current_data_row['Close'], current_timestamp, "NaN data in features")
                        else:
                            logger.warning(f"Cannot close trade for {ticker} at {current_timestamp} due to missing 'Close' price.")
                    continue # Skip this ticker for this timestamp

                current_price = current_data_row['Close']

                # --- Position Management (Exit Logic for open trades) ---
                if self.open_trades[ticker] is not None:
                    trade = self.open_trades[ticker]

                    # Check Stop Loss
                    if (trade['direction'] == 'Long' and current_price <= trade['stop_loss_level']) or \
                       (trade['direction'] == 'Short' and current_price >= trade['stop_loss_level']):
                        self._close_trade(ticker, current_price, current_timestamp, "Stop Loss")
                        continue # Move to next ticker or next time step (after closing trade)

                    # Check Take Profit
                    if (trade['direction'] == 'Long' and current_price >= trade['take_profit_level']) or \
                       (trade['direction'] == 'Short' and current_price <= trade['take_profit_level']):
                        self._close_trade(ticker, current_price, current_timestamp, "Take Profit")
                        continue # Move to next ticker or next time step (after closing trade)
                
                # --- Entry Logic (if no position is open) ---
                if self.open_trades[ticker] is None:
                    # Calculate desired units using the strategy's get_my_position
                    desired_units = self.strategy.get_my_position(current_data_row, prev_data_row, ticker)
                    
                    if desired_units != 0:
                        # Determine direction and open trade
                        if desired_units > 0: # Go Long
                            self._open_trade(ticker, current_price, current_timestamp, 'Long', desired_units)
                        elif desired_units < 0: # Go Short
                            self._open_trade(ticker, current_price, current_timestamp, 'Short', desired_units)
            
            # End of current timestamp loop for all tickers
            # The equity curve is updated at the beginning of each period.
            # No need for explicit portfolio value calculation here, as trades modify self.current_capital.

        # After loop, close any remaining open positions at the final price
        final_timestamp = backtest_dates.max()
        final_data_row_overall = self.df_full_features.loc[final_timestamp] # Get last row of combined features
        
        # Append final capital to equity curve
        self.equity_curve_data.append({'Time': final_timestamp, 'Equity': self.current_capital}) 

        for ticker in list(self.open_trades.keys()): # Iterate over a copy as dict might change
            if self.open_trades[ticker] is not None:
                final_price_for_ticker = final_data_row_overall[ticker]['Close']
                self._close_trade(ticker, final_price_for_ticker, final_timestamp, "End of Backtest")

        logger.info("Backtest simulation completed.")

    def _open_trade(self, ticker, entry_price, entry_time, direction, units):
        """Helper to record opening a trade."""
        
        # Calculate transaction costs and slippage for opening
        cost_open = self.calculate_transaction_costs(0, units, entry_price) # Old units 0, new units 'units'
        slippage_open = self.apply_slippage(abs(units), entry_price)
        total_cost = cost_open + slippage_open

        self.current_capital -= total_cost # Deduct commission and slippage from capital
        
        # Define TP/SL levels using the actual High/Low from current_data_row if available, otherwise fixed percentage
        # Note: current_data_row is available from the main loop if needed, but for simplicity, using fixed percentage
        # as per previous discussion, or using the LOOKBACK_PERIOD values if they are reliably calculated.
        # Since feature_engineering now calculates High_LOOKBACK_PERIOD_Period and Low_LOOKBACK_PERIOD_Period,
        # we can use them IF they are not NaN.
        
        # Retrieve the current data row to get High/Low_Period values
        current_data_row_for_tp_sl = self.df_full_features.loc[entry_time, ticker]
        
        if direction == 'Long':
            stop_loss_level = entry_price * (1 - STOP_LOSS_PCT)
            # Use High_LOOKBACK_PERIOD_Period for TP if available and not NaN, else use fixed percentage
            if f'High_{LOOKBACK_PERIOD}_Period' in current_data_row_for_tp_sl and not pd.isna(current_data_row_for_tp_sl[f'High_{LOOKBACK_PERIOD}_Period']):
                take_profit_level = current_data_row_for_tp_sl[f'High_{LOOKBACK_PERIOD}_Period']
            else:
                take_profit_level = entry_price * (1 + STOP_LOSS_PCT) # Fallback to fixed percentage
        else: # Short
            stop_loss_level = entry_price * (1 + STOP_LOSS_PCT)
            # Use Low_LOOKBACK_PERIOD_Period for TP if available and not NaN, else use fixed percentage
            if f'Low_{LOOKBACK_PERIOD}_Period' in current_data_row_for_tp_sl and not pd.isna(current_data_row_for_tp_sl[f'Low_{LOOKBACK_PERIOD}_Period']):
                take_profit_level = current_data_row_for_tp_sl[f'Low_{LOOKBACK_PERIOD}_Period']
            else:
                take_profit_level = entry_price * (1 - STOP_LOSS_PCT) # Fallback to fixed percentage

        self.open_trades[ticker] = {
            'entry_price': entry_price,
            'entry_time': entry_time,
            'direction': direction,
            'units': units, # Store the units for this trade
            'stop_loss_level': stop_loss_level,
            'take_profit_level': take_profit_level
        }
        logger.info(f"Opened {direction} trade for {ticker} at {entry_price:.2f} on {entry_time}. Units: {units:.2f}. Capital: {self.current_capital:.2f}. SL: {stop_loss_level:.2f}, TP: {take_profit_level:.2f}")

    def _close_trade(self, ticker, exit_price, exit_time, reason):
        """Helper to record closing a trade and update capital."""
        trade = self.open_trades[ticker]
        if trade is None: # Should not happen if logic is correct, but for safety
            logger.warning(f"Attempted to close a non-existent trade for {ticker} at {exit_time}.")
            return

        if trade['direction'] == 'Long':
            profit_loss = (exit_price - trade['entry_price']) * trade['units']
        else: # Short
            profit_loss = (trade['entry_price'] - exit_price) * abs(trade['units']) # For short, units are negative, so use abs
        
        # Calculate transaction costs and slippage for closing
        cost_close = self.calculate_transaction_costs(trade['units'], 0, exit_price) # Old units 'units', new units 0
        slippage_close = self.apply_slippage(abs(trade['units']), exit_price)
        total_cost = cost_close + slippage_close

        self.current_capital += profit_loss # Add/subtract profit/loss
        self.current_capital -= total_cost # Deduct exit commission and slippage
        
        profit_loss_pct = profit_loss / (trade['entry_price'] * trade['units']) if (trade['entry_price'] * trade['units']) != 0 else 0
        
        self.trade_log_data.append({
            'Ticker': ticker,
            'Entry_Time': trade['entry_time'],
            'Exit_Time': exit_time,
            'Direction': trade['direction'],
            'Entry_Price': trade['entry_price'],
            'Exit_Price': exit_price,
            'Units': trade['units'],
            'Profit_Loss_Abs': profit_loss,
            'Profit_Loss_Pct': profit_loss_pct,
            'Reason': reason
        })
        logger.info(f"Closed {trade['direction']} trade for {ticker} at {exit_price:.2f} on {exit_time}. P/L: {profit_loss:.2f} ({profit_loss_pct:.4f}). Reason: {reason}. Capital: {self.current_capital:.2f}")
        self.open_trades[ticker] = None # Mark trade as closed

    def get_equity_curve(self) -> pd.Series:
        """Returns the equity curve as a pandas Series."""
        if not self.equity_curve_data:
            return pd.Series(dtype=float)
        df_equity = pd.DataFrame(self.equity_curve_data).set_index('Time')
        return df_equity['Equity']

    def get_trade_log(self) -> pd.DataFrame:
        """Returns the trade log as a pandas DataFrame."""
        return pd.DataFrame(self.trade_log_data)

    def calculate_performance_metrics(self, results_df: pd.DataFrame):
        """
        Calculate comprehensive performance metrics from the backtest results DataFrame.
        
        Args:
            results_df (pd.DataFrame): DataFrame containing 'portfolio_value' and 'daily_return'.
            
        Returns:
            dict: Performance metrics.
        """
        returns = results_df['daily_return'].dropna()
        portfolio_values = results_df['portfolio_value']
        
        if portfolio_values.empty or len(portfolio_values) < 2:
            logger.warning("Not enough data in results_df to calculate full performance metrics.")
            return {}

        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        
        # Annualized return (assuming 252 trading days/year for daily data)
        # If interval is 5m, there are many more periods per day.
        # Let's calculate the number of trading periods per year based on the interval
        # Assuming 6.5 trading hours per day (9:30 AM to 4:00 PM ET)
        trading_hours_per_day = 6.5
        # Extract numeric part from INTERVAL (e.g., '5m' -> 5, '1h' -> 1)
        interval_value = int(''.join(filter(str.isdigit, INTERVAL)))
        if 'm' in INTERVAL:
            periods_per_hour = 60 / interval_value
        elif 'h' in INTERVAL:
            periods_per_hour = 1 / interval_value # 1 period per hour if interval is 1h
        else: # Assume daily if no unit specified
            periods_per_hour = 1 # Not really periods per hour, but 1 period per day
            trading_hours_per_day = 1 # Adjust for daily calculation

        periods_per_day_calc = trading_hours_per_day * periods_per_hour
        
        # Total periods in the backtest
        num_periods = len(returns)
        
        # Annualization factor
        annualization_factor = (periods_per_day_calc * 252) / num_periods if num_periods > 0 else 1
        
        annualized_return = (1 + total_return) ** annualization_factor - 1 if total_return > -1 else -1
        
        # Volatility
        volatility = returns.std() * np.sqrt(periods_per_day_calc * 252) # Annualized volatility

        # Risk metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # Additional metrics from trade log
        trade_log_df = self.get_trade_log()
        total_trades = trade_log_df.shape[0] if not trade_log_df.empty else 0
        
        win_rate = (trade_log_df['Profit_Loss_Pct'] > 0).mean() if total_trades > 0 else 0
        avg_win = trade_log_df[trade_log_df['Profit_Loss_Pct'] > 0]['Profit_Loss_Pct'].mean() if (trade_log_df['Profit_Loss_Pct'] > 0).any() else 0
        avg_loss = trade_log_df[trade_log_df['Profit_Loss_Pct'] < 0]['Profit_Loss_Pct'].mean() if (trade_log_df['Profit_Loss_Pct'] < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Total transaction costs (sum from trade log)
        # This is the most accurate way to sum up all costs explicitly recorded.
        # Sum of costs from _open_trade and _close_trade.
        total_costs_incurred = sum(trade['total_cost_open'] for trade in self.trade_log_data) + \
                               sum(trade['total_cost_close'] for trade in self.trade_log_data)
        
        cost_ratio = total_costs_incurred / self.initial_capital if self.initial_capital != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'total_transaction_costs': total_costs_incurred,
            'cost_ratio': cost_ratio,
            'num_trades': total_trades,
            'final_portfolio_value': portfolio_values.iloc[-1]
        }
        
    def _calculate_max_drawdown(self, portfolio_values: pd.Series):
        """
        Calculate maximum drawdown.
        
        Args:
            portfolio_values (pd.Series): Portfolio values over time.
            
        Returns:
            float: Maximum drawdown (negative value).
        """
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min()
        
        return max_drawdown
        
    def plot_results(self, save_plots=SAVE_PLOTS):
        """
        Plot backtest results.
        """
        if not self.backtest_results:
            logger.error("No backtest results to plot. Run run_backtest() first.")
            return
            
        results_df = self.backtest_results['results_df']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Backtesting Results', fontsize=16)
        
        # Portfolio value over time
        axes[0, 0].plot(results_df['date'], results_df['portfolio_value'], linewidth=2)
        axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.2f')) # Currency format

        # Cumulative returns
        axes[0, 1].plot(results_df['date'], results_df['cumulative_return'] * 100, linewidth=2, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Cumulative Returns (%)')
        axes[0, 1].set_ylabel('Cumulative Return (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].yaxis.set_major_formatter(mticker.PercentFormatter()) # Percent format
        
        # Daily returns histogram
        sns.histplot(results_df['daily_return'].dropna() * 100, kde=True, ax=axes[1, 0], color='skyblue')
        axes[1, 0].set_title('Daily Returns Distribution (%)')
        axes[1, 0].set_xlabel('Daily Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Daily transaction costs
        axes[1, 1].plot(results_df['date'], results_df['transaction_costs'], linewidth=1, color='purple', alpha=0.7)
        axes[1, 1].set_title('Daily Transaction Costs')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Costs ($)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.2f')) # Currency format
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        
        if save_plots:
            plot_file = RESULTS_DIR / 'plots' / f"backtest_results_{INTERVAL}.png"
            plt.savefig(plot_file)
            logger.info(f"Saved backtest plots to {plot_file}")
        plt.show()

    def display_performance_summary(self):
        """
        Display the calculated performance metrics.
        """
        if not self.backtest_results:
            logger.error("No backtest results to display. Run run_backtest() first.")
            return
        
        metrics = self.backtest_results['performance_metrics']
        params = self.backtest_results['parameters']

        print("\n" + "="*80)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Initial Capital: ${params['initial_capital']:.2f}")
        print(f"Final Portfolio Value: ${metrics['final_portfolio_value']:.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Annualized Volatility: {metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Average Winning Trade (%): {metrics['avg_win_pct']:.2%}")
        print(f"Average Losing Trade (%): {metrics['avg_loss_pct']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Total Trades: {metrics['num_trades']}")
        print(f"Total Transaction Costs: ${metrics['total_transaction_costs']:.2f}")
        print(f"Cost Ratio (Total Costs / Initial Capital): {metrics['cost_ratio']:.2%}")
        print("="*80)


def main():
    """
    Main function to run the backtester.
    """
    print("Initializing Backtester...")
    
    try:
        backtester = Backtester()
        
        # Run the backtest
        backtest_results = backtester.run_backtest()
        
        if backtest_results:
            # Store results in the backtester instance for plotting/summary
            backtester.backtest_results = backtest_results

            # Save results DataFrame
            results_df_path = RESULTS_DIR / 'data' / f"backtest_results_df_{INTERVAL}.csv"
            backtest_results['results_df'].to_csv(results_df_path, index=False)
            logger.info(f"Backtest results DataFrame saved to {results_df_path}")

            # Display summary and plot
            backtester.display_performance_summary()
            backtester.plot_results()
        else:
            print("Backtest failed or returned no results.")
        
        print("\nBacktesting process completed.")
        
    except Exception as e:
        logger.error(f"Error during backtesting process: {e}")
        print(f"An error occurred during backtesting: {e}")

if __name__ == "__main__":
    main()

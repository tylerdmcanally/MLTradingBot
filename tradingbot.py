import os
import pandas as pd
from dotenv import load_dotenv
from datetime import timedelta
from finbert_utils import estimate_sentiment

from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from lumibot.strategies.strategy import Strategy
from alpaca.trading.client import TradingClient # Import the TradingClient

# Load environment variables from .env file
load_dotenv()

# --- Alpaca API Configuration from .env file ---
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# --- Lumibot Broker Setup ---
ALPACA_CONFIG = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True,
}

def get_sp500_tickers():
    """Scrapes the S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    tickers = table[0]['Symbol'].tolist()
    # The symbol 'BF.B' can be problematic for some APIs, so we remove it.
    if 'BF.B' in tickers:
        tickers.remove('BF.B')
    return tickers

class MLTrader(Strategy):
    def initialize(self, cash_per_trade: float = 0.1):
        """Initialize the strategy with cash allocation per trade."""
        self.sleeptime = "24H"
        self.symbols = get_sp500_tickers()
        self.log_message(f"Successfully scraped {len(self.symbols)} tickers.")
        self.cash_per_trade = cash_per_trade
        
        self.log_message("Getting data API from broker...")
        # Create a dedicated client for getting news data directly.
        self.api = TradingClient(API_KEY, API_SECRET, paper=True)
        self.log_message("Successfully got data API.", color="green")
        
        # --- Pre-load the sentiment model ---
        self.log_message("Initializing sentiment model (one-time download)...", color="yellow")
        self.log_message("This may take several minutes. Please be patient.", color="yellow")
        try:
            estimate_sentiment(["test"]) # A dummy call to trigger the download
            self.log_message("Sentiment model initialized successfully.", color="green")
        except Exception as e:
            self.log_message(f"Error initializing sentiment model: {e}", color="red")
            self.log_message("Bot will not be able to analyze sentiment. Stopping.", color="red")
            self.deactivate()

        self.log_message(f"Bot initialized. Tracking {len(self.symbols)} S&P 500 stocks.")

    def on_trading_iteration(self):
        """
        Gathers all buy signals, ranks them, and executes trades
        based on available cash and allocation rules.
        """
        # --- 1. Signal Gathering Phase ---
        buy_candidates = []
        positions = self.get_positions()
        owned_symbols = [p.symbol for p in positions]

        self.log_message("Gathering buy signals for all symbols...")
        
        # Add a progress counter
        i = 0
        for symbol in self.symbols:
            # Log progress every 25 stocks
            i += 1
            if (i % 25 == 0):
                self.log_message(f"Analyzing... ({i}/{len(self.symbols)})")

            if symbol in owned_symbols:
                continue

            probability, sentiment = self.get_sentiment_for_symbol(symbol)
            if sentiment == "positive" and probability > 0.999:
                buy_candidates.append({"symbol": symbol, "probability": probability})
                self.log_message(f"Buy candidate found: {symbol} (Prob: {probability:.4f})", color="blue")

        if not buy_candidates:
            self.log_message("No strong buy signals found in this iteration.", color="yellow")
            return

        # --- 2. Ranking Phase ---
        ranked_candidates = sorted(buy_candidates, key=lambda x: x['probability'], reverse=True)
        self.log_message(f"Found {len(ranked_candidates)} potential buy candidates. Ranking complete.")

        # --- 3. Execution Phase ---
        cash = self.get_cash()
        self.log_message(f"Available cash for trading: ${cash:,.2f}")

        for candidate in ranked_candidates:
            symbol = candidate['symbol']
            
            cash_for_this_trade = cash * self.cash_per_trade
            last_price = self.get_last_price(symbol)

            if last_price > 0:
                quantity = round(cash_for_this_trade / last_price, 0)
                cost = quantity * last_price

                if quantity > 0 and cash >= cost:
                    order = self.create_order(
                        symbol,
                        quantity,
                        "buy",
                        type="bracket",
                        take_profit_price=last_price * 1.20,
                        stop_loss_price=last_price * 0.95
                    )
                    self.submit_order(order)
                    self.log_message(f"Submitted buy order for {quantity} shares of {symbol}", color="green")
                    cash -= cost
                else:
                    self.log_message("Insufficient cash for further trades. Ending execution phase.", color="yellow")
                    break

    def get_sentiment_for_symbol(self, symbol: str):
        """Gets sentiment for a single symbol, with error handling."""
        today = self.get_datetime()
        three_days_prior = today - timedelta(days=3)
        try:
            news = self.api.get_news(symbol=symbol,
                                     start=three_days_prior.strftime('%Y-%m-%d'),
                                     end=today.strftime('%Y-%m-%d'))
            if not news:
                return 0, "neutral"
            headlines = [ev.__dict__["_raw"]["headline"] for ev in news]
            probability, sentiment = estimate_sentiment(headlines)
            return probability, sentiment
        except Exception as e:
            # Log the error but don't crash the bot
            self.log_message(f"Error getting news for {symbol}: {e}", color="red")
            return 0, "neutral"

# --- Main Execution Block ---
if __name__ == "__main__":
    trader = Trader()
    strategy = MLTrader(
        name='ml_strat_stratified',
        broker=Alpaca(ALPACA_CONFIG),
        parameters={"cash_per_trade": 0.1}
    )
    trader.add_strategy(strategy)
    trader.run_all()

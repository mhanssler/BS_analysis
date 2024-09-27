import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime

# Black-Scholes formula for European options
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the theoretical price of European call or put options using the Black-Scholes formula.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put option
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Calculate Greeks
def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the Greeks for European call or put options.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma **2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type=='call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - \
            r * K * np.exp(-r*T) * norm.cdf(d2 if option_type=='call' else -d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r*T) * norm.cdf(d2 if option_type=='call' else -d2)
    return delta, gamma, theta, vega, rho

# Fetch option data
def get_option_data(ticker):
    """
    Fetches current stock price and option chain data for a given ticker symbol.
    """
    stock = yf.Ticker(ticker)
    S = stock.history(period='1d')['Close'][0]
    expirations = stock.options
    options_data = []
    for expiration in expirations:
        options = stock.option_chain(expiration)
        options_data.append({
            'expiration': expiration,
            'calls': options.calls,
            'puts': options.puts
        })
    return S, options_data

# Main function
def main():
    ticker = input("Enter the ticker symbol (e.g., AAPL): ").upper()
    S, options_data = get_option_data(ticker)
    print(f"\nCurrent stock price (S): ${S:.2f}")

    # Estimate historical volatility (sigma)
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1y')
    hist['Returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
    sigma = hist['Returns'].std() * np.sqrt(252)  # Annualized volatility
    print(f"Estimated volatility (sigma): {sigma:.2%}")

    # Risk-free interest rate (e.g., current 1-year Treasury rate)
    r = 0.05  # 5% annual interest rate
    print(f"Assumed risk-free interest rate (r): {r:.2%}")

    # Mispricing threshold
    threshold = float(input("Enter the mispricing threshold (e.g., 0.50 for $0.50): "))

    # Process each expiration date
    for data in options_data:
        expiration = data['expiration']
        T = (datetime.strptime(expiration, '%Y-%m-%d') - datetime.now()).days / 365.0
        if T <= 0:
            continue  # Skip expired options
        print(f"\nOptions expiring on {expiration} ({T*365:.0f} days to expiration)")

        # Process calls and puts
        for option_type in ['calls', 'puts']:
            print(f"\n{option_type.capitalize()}:")
            options = data[option_type]
            # Calculate theoretical prices and Greeks
            options['Theoretical'] = options.apply(
                lambda row: black_scholes(S, row['strike'], T, r, sigma, option_type[:-1]), axis=1)
            options['Difference'] = options['lastPrice'] - options['Theoretical']
            options['Delta'], options['Gamma'], options['Theta'], options['Vega'], options['Rho'] = zip(*options.apply(
                lambda row: calculate_greeks(S, row['strike'], T, r, sigma, option_type[:-1]), axis=1))

            # Identify mispriced options
            mispriced = options[np.abs(options['Difference']) >= threshold]
            if not mispriced.empty:
                print(f"\nMispriced {option_type} options (Difference >= ${threshold}):")
                print(mispriced[['contractSymbol', 'strike', 'lastPrice', 'Theoretical', 'Difference',
                                 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho']].round(2).to_string(index=False))
            else:
                print(f"No mispriced {option_type} options found with the given threshold.")

if __name__ == "__main__":
    main()

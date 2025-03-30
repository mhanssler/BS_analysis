import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

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
    try:
        # Try multiple periods to get the stock price
        stock = yf.Ticker(ticker)
        for period in ['1d', '5d', '1mo']:
            try:
                history = stock.history(period=period)
                if not history.empty:
                    S = history['Close'].iloc[-1]
                    print(f"Successfully fetched price data for {ticker}")
                    break
            except Exception as e:
                continue
        else:
            raise ValueError(f"Could not fetch price data for {ticker} after multiple attempts")

        # Get options data with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                expirations = stock.options
                if not expirations:
                    raise ValueError(f"No options data found for ticker {ticker}")
                
                options_data = []
                for expiration in expirations:
                    try:
                        options = stock.option_chain(expiration)
                        options_data.append({
                            'expiration': expiration,
                            'calls': options.calls,
                            'puts': options.puts
                        })
                        print(f"Successfully fetched options data for expiration {expiration}")
                    except Exception as e:
                        print(f"Warning: Could not fetch options data for expiration {expiration}: {str(e)}")
                        continue
                
                if options_data:
                    return S, options_data
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    continue
                else:
                    raise ValueError(f"Could not fetch options data after {max_retries} attempts")
        
        raise ValueError(f"No valid options data found for {ticker}")
    
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {str(e)}")

def analyze_option(options_df, option_type, S, T, r, sigma):
    """
    Analyzes options and provides recommendations based on various metrics.
    """
    # Calculate additional metrics
    options_df['Intrinsic_Value'] = options_df.apply(
        lambda row: max(0, S - row['strike']) if option_type == 'call' else max(0, row['strike'] - S), axis=1)
    options_df['Time_Value'] = options_df['lastPrice'] - options_df['Intrinsic_Value']
    options_df['IV_Ratio'] = options_df['Intrinsic_Value'] / options_df['lastPrice']
    options_df['Time_Value_Ratio'] = options_df['Time_Value'] / options_df['lastPrice']
    
    # Calculate implied volatility
    options_df['Implied_Vol'] = options_df.apply(
        lambda row: sigma * (1 + row['Difference'] / row['lastPrice']), axis=1)
    
    # Calculate risk metrics
    options_df['Risk_Score'] = options_df.apply(
        lambda row: abs(row['Delta']) * (1 + abs(row['Theta']) / 365), axis=1)
    
    return options_df

def get_option_recommendations(options_df, option_type, threshold):
    """
    Generates recommendations for options based on analysis.
    """
    recommendations = []
    
    # Sort by risk-adjusted potential return
    options_df['Potential_Return'] = options_df['Difference'] / options_df['Risk_Score']
    sorted_options = options_df.sort_values('Potential_Return', ascending=False)
    
    # Get top 3 recommendations
    for _, option in sorted_options.head(3).iterrows():
        if abs(option['Difference']) >= threshold:
            recommendation = {
                'contract': option['contractSymbol'],
                'strike': option['strike'],
                'price': option['lastPrice'],
                'theoretical': option['Theoretical'],
                'difference': option['Difference'],
                'delta': option['Delta'],
                'theta': option['Theta'],
                'risk_score': option['Risk_Score'],
                'reasoning': []
            }
            
            # Generate reasoning
            if option['Difference'] > 0:
                recommendation['reasoning'].append("Option appears undervalued compared to theoretical price")
            else:
                recommendation['reasoning'].append("Option appears overvalued compared to theoretical price")
            
            if abs(option['Delta']) > 0.7:
                recommendation['reasoning'].append("High delta indicates strong price sensitivity")
            elif abs(option['Delta']) < 0.3:
                recommendation['reasoning'].append("Low delta indicates limited price sensitivity")
            
            if option['Theta'] < -0.1:
                recommendation['reasoning'].append("High time decay - consider shorter-term holding period")
            
            if option['Risk_Score'] > 1.5:
                recommendation['reasoning'].append("High risk score - position sizing should be conservative")
            
            recommendations.append(recommendation)
    
    return recommendations

def print_recommendations(recommendations, option_type):
    """
    Prints formatted recommendations for options.
    """
    if not recommendations:
        print(f"\nNo strong recommendations found for {option_type} options.")
        return
    
    print(f"\nTop {option_type.capitalize()} Recommendations:")
    print("=" * 80)
    for i, rec in enumerate(recommendations, 1):
        print(f"\nRecommendation {i}:")
        print(f"Contract: {rec['contract']}")
        print(f"Strike: ${rec['strike']:.2f}")
        print(f"Current Price: ${rec['price']:.2f}")
        print(f"Theoretical Price: ${rec['theoretical']:.2f}")
        print(f"Price Difference: ${rec['difference']:.2f}")
        print(f"Delta: {rec['delta']:.2f}")
        print(f"Theta: {rec['theta']:.2f}")
        print(f"Risk Score: {rec['risk_score']:.2f}")
        print("\nReasoning:")
        for reason in rec['reasoning']:
            print(f"- {reason}")
        print("-" * 80)

# Main function
def main():
    while True:
        try:
            ticker = input("Enter the ticker symbol (e.g., AAPL): ").upper()
            S, options_data = get_option_data(ticker)
            break
        except ValueError as e:
            print(f"Error: {e}")
            print("Please try another ticker symbol.")
            continue
    
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
            
            # Analyze options and get recommendations
            options = analyze_option(options, option_type[:-1], S, T, r, sigma)
            recommendations = get_option_recommendations(options, option_type[:-1], threshold)
            print_recommendations(recommendations, option_type[:-1])

if __name__ == "__main__":
    main()

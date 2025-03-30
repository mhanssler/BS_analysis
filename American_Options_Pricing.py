import yfinance as yf
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import time
from tqdm import tqdm

def check_gpu_availability():
    """
    Checks if CUDA is available and returns the device to use.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def binomial_tree_american(S, K, T, r, sigma, option_type='call', n_steps=100, device=None, timeout=30):
    """
    Calculates the theoretical price of American call or put options using the Binomial Tree model.
    Supports GPU acceleration when available.
    """
    start_time = time.time()
    if device is None:
        device = check_gpu_availability()
    
    # Calculate tree parameters
    dt = T / n_steps
    u = torch.exp(torch.tensor(sigma * np.sqrt(dt), device=device))
    d = 1 / u
    p = (torch.exp(torch.tensor(r * dt, device=device)) - d) / (u - d)
    
    # Initialize price tree
    price_tree = torch.zeros((n_steps + 1, n_steps + 1), device=device)
    price_tree[0, 0] = S
    
    # Build price tree
    for i in range(1, n_steps + 1):
        if time.time() - start_time > timeout:
            raise TimeoutError("Calculation timed out")
        price_tree[i, 0] = price_tree[i-1, 0] * u
        for j in range(1, i + 1):
            price_tree[i, j] = price_tree[i-1, j-1] * d
    
    # Initialize option value tree
    option_tree = torch.zeros((n_steps + 1, n_steps + 1), device=device)
    
    # Calculate terminal values
    if option_type == 'call':
        option_tree[n_steps, :] = torch.maximum(price_tree[n_steps, :] - K, torch.tensor(0., device=device))
    else:  # put option
        option_tree[n_steps, :] = torch.maximum(K - price_tree[n_steps, :], torch.tensor(0., device=device))
    
    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        if time.time() - start_time > timeout:
            raise TimeoutError("Calculation timed out")
        for j in range(i + 1):
            # Calculate expected value
            expected_value = torch.exp(torch.tensor(-r * dt, device=device)) * (p * option_tree[i + 1, j] + (1 - p) * option_tree[i + 1, j + 1])
            
            # For American options, compare with immediate exercise value
            if option_type == 'call':
                immediate_value = torch.maximum(price_tree[i, j] - K, torch.tensor(0., device=device))
            else:  # put option
                immediate_value = torch.maximum(K - price_tree[i, j], torch.tensor(0., device=device))
            
            option_tree[i, j] = torch.maximum(expected_value, immediate_value)
    
    return option_tree[0, 0].item()

def calculate_greeks_american(S, K, T, r, sigma, option_type='call', n_steps=100, device=None):
    """
    Calculates the Greeks for American options using finite differences.
    Supports GPU acceleration when available.
    """
    if device is None:
        device = check_gpu_availability()
    
    # Calculate base price
    base_price = binomial_tree_american(S, K, T, r, sigma, option_type, n_steps, device)
    
    # Delta
    delta = (binomial_tree_american(S * 1.01, K, T, r, sigma, option_type, n_steps, device) - 
             binomial_tree_american(S * 0.99, K, T, r, sigma, option_type, n_steps, device)) / (0.02 * S)
    
    # Gamma
    gamma = (binomial_tree_american(S * 1.01, K, T, r, sigma, option_type, n_steps, device) + 
             binomial_tree_american(S * 0.99, K, T, r, sigma, option_type, n_steps, device) - 
             2 * base_price) / (0.0001 * S * S)
    
    # Theta
    theta = (binomial_tree_american(S, K, T - 1/365, r, sigma, option_type, n_steps, device) - 
             base_price) / (1/365)
    
    # Vega
    vega = (binomial_tree_american(S, K, T, r, sigma * 1.01, option_type, n_steps, device) - 
            binomial_tree_american(S, K, T, r, sigma * 0.99, option_type, n_steps, device)) / (0.02 * sigma)
    
    # Rho
    rho = (binomial_tree_american(S, K, T, r * 1.01, sigma, option_type, n_steps, device) - 
           binomial_tree_american(S, K, T, r * 0.99, sigma, option_type, n_steps, device)) / (0.02 * r)
    
    return delta, gamma, theta, vega, rho

# Reuse the same data fetching and analysis functions from European_Options_Pricing.py
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

def main():
    # Check GPU availability and ask user preference
    device = check_gpu_availability()
    if device.type == "cuda":
        use_gpu = input("GPU acceleration is available. Would you like to use it? (y/n): ").lower() == 'y'
        if not use_gpu:
            device = torch.device("cpu")
        print(f"Using {'GPU' if use_gpu else 'CPU'} for calculations")
    else:
        print("GPU acceleration is not available. Using CPU for calculations.")
        device = torch.device("cpu")

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
    
    # Maximum number of options to process per expiration
    max_options = int(input("Enter maximum number of options to process per expiration (e.g., 20): "))

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
            
            if len(options) == 0:
                print(f"No {option_type} available for this expiration date.")
                continue
            
            # Limit the number of options to process
            if len(options) > max_options:
                # Select options closest to the current stock price
                options = options.iloc[(options['strike'] - S).abs().argsort()[:max_options]]
                print(f"Processing {max_options} options closest to current price")
            
            # Calculate theoretical prices and Greeks using American option model
            print("Calculating theoretical prices...")
            options['Theoretical'] = options.apply(
                lambda row: binomial_tree_american(S, row['strike'], T, r, sigma, option_type[:-1], device=device), axis=1)
            options['Difference'] = options['lastPrice'] - options['Theoretical']
            
            # Calculate Greeks with progress bar
            print("Calculating Greeks...")
            greeks_list = []
            for _, row in tqdm(options.iterrows(), total=len(options), desc="Processing options"):
                try:
                    greeks = calculate_greeks_american(S, row['strike'], T, r, sigma, option_type[:-1], device=device)
                    greeks_list.append(greeks)
                except Exception as e:
                    print(f"\nWarning: Could not calculate Greeks for strike {row['strike']}: {str(e)}")
                    greeks_list.append((0.0, 0.0, 0.0, 0.0, 0.0))  # Default values for Delta, Gamma, Theta, Vega, Rho
            
            if greeks_list:
                options['Delta'], options['Gamma'], options['Theta'], options['Vega'], options['Rho'] = zip(*greeks_list)
            else:
                for greek in ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']:
                    options[greek] = 0.0
            
            # Analyze options and get recommendations
            options = analyze_option(options, option_type[:-1], S, T, r, sigma)
            recommendations = get_option_recommendations(options, option_type[:-1], threshold)
            print_recommendations(recommendations, option_type[:-1])

if __name__ == "__main__":
    main() 
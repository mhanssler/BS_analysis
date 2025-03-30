# Options Analysis Tool

A Python-based tool for analyzing options using both Black-Scholes (European) and Binomial Tree (American) models, providing trading recommendations based on various metrics.

## Features

- Real-time options data fetching using yfinance
- Support for both European and American options
- Black-Scholes model implementation for European options
- Binomial Tree model implementation for American options
- Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Risk analysis and scoring
- Automated recommendations based on multiple factors
- Support for both calls and puts
- Multiple expiration date analysis

## Prerequisites

- Python 3.11 or higher
- Poetry for dependency management

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BS_analysis
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Usage

The tool provides two separate scripts for analyzing European and American options:

### European Options Analysis
```bash
poetry run python European_Options_Pricing.py
```

### American Options Analysis
```bash
poetry run python American_Options_Pricing.py
```

For both scripts:
1. Enter the ticker symbol when prompted (e.g., AAPL, MSFT, GOOGL)
2. Enter your mispricing threshold (e.g., 0.50 for $0.50)

## Output Explanation

The scripts provide detailed analysis including:

### Basic Information
- Current stock price
- Estimated volatility
- Risk-free interest rate

### For Each Expiration Date
- Top 3 recommendations for calls
- Top 3 recommendations for puts
- Detailed metrics for each recommendation:
  - Contract details
  - Strike price
  - Current price
  - Theoretical price
  - Price difference
  - Greeks (Delta, Theta)
  - Risk score
  - Reasoning for the recommendation

## Key Metrics

### Greeks
- **Delta**: Price sensitivity to underlying stock
- **Theta**: Time decay
- **Gamma**: Delta sensitivity
- **Vega**: Volatility sensitivity
- **Rho**: Interest rate sensitivity

### Risk Metrics
- **Risk Score**: Combines Delta and Theta
- **Intrinsic Value**: Current value if exercised
- **Time Value**: Premium above intrinsic value

### Recommendation Criteria
- Price mispricing (difference from theoretical)
- Risk-adjusted potential return
- Greeks analysis
- Risk metrics

## Important Notes

1. The scripts use historical volatility for calculations
2. Risk-free rate is set to 5% (can be modified in the code)
3. Recommendations are based on:
   - Price mispricing
   - Risk-adjusted returns
   - Greeks analysis
   - Risk metrics

4. Always verify the data and do your own research before making trading decisions

## Dependencies

- yfinance: For fetching market data
- pandas: For data manipulation
- numpy: For numerical computations
- scipy: For statistical functions

## License

MIT License

## Disclaimer

This tool is for educational and research purposes only. It should not be used as the sole basis for making investment decisions. Always conduct your own research and consider consulting with financial advisors before making investment decisions. 
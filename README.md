# Transformer-based-Richard-Wyckoff-Chatbot

# Wyckoff Trading Assistant

A comprehensive web application that combines Reinforcement Learning (RL) with Wyckoff methodology principles to assist traders in analyzing stocks and making informed trading decisions.

## Overview

Wyckoff Trading Assistant offers two main features:
1. **Wyckoff AI Assistant**: Ask questions about Wyckoff methodology, market structure, and trading principles
2. **RL-Enhanced Trading Strategy**: Backtest a Q-learning based trading strategy on any stock

The application uses a reinforcement learning algorithm to learn optimal trading strategies based on historical price data, combined with an AI chatbot trained on Wyckoff methodology principles.

## Features

### Wyckoff AI Assistant
- Interactive chat interface for questions about Wyckoff methodology
- Trained on comprehensive Wyckoff trading principles and concepts
- Answers questions about springs, upthrusts, accumulation, distribution, and more
- Helps traders understand market structure and price action

### RL Trading Strategy
- Q-learning based trading algorithm that learns optimal Buy/Sell/Hold decisions
- Customizable backtesting on any stock symbol
- Performance metrics including ROI, portfolio value, and trade signals
- Visual representation of trading signals overlaid on price charts

### Interactive Dashboard
- Real-time stock data visualization
- Technical indicators (MA20, MA50, MA200, RSI, Bollinger Bands)
- Performance metrics for strategy evaluation
- Trade signal visualization

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Machine Learning**: PyTorch (for the Wyckoff chatbot), Q-learning (for the trading strategy)
- **Data**: yfinance for real-time and historical stock data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/wyckoff-trading-assistant.git
cd wyckoff-trading-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv wyckoff_venv
# On Windows
wyckoff_venv\Scripts\activate
# On macOS/Linux
source wyckoff_venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
You can download the model file fromk here()

4. Make sure you have the model file:
- Place your transformer model file (`transformer_chatbot_gpu_deco_2.pth`) in the correct location
- Update the path in `app.py` if needed

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to:
```
http://localhost:5000
```

## Reinforcement Learning Trading Strategy

The trading strategy uses Q-learning, a model-free reinforcement learning algorithm that learns a policy to maximize expected future rewards. The implementation includes:

### Environment (StockTradingEnv)
- **State**: Current stock price and position (holding/not holding)
- **Actions**: Buy, Sell, Hold
- **Rewards**: Changes in portfolio value

### Q-Learning Algorithm
- **Q-Table**: Stores state-action values
- **Epsilon-Greedy Policy**: Balances exploration and exploitation
- **Learning Rate (Alpha)**: Controls adaptation speed
- **Discount Factor (Gamma)**: Weighs importance of future rewards

### State Discretization
- Continuous price values are mapped to discrete bins
- Enables tabular representation for the Q-learning algorithm

### Backtest Procedure
1. Historical data is fetched for the specified stock and time period
2. The environment is initialized with the data
3. Q-learning algorithm trains on the data to learn optimal policy
4. The learned policy is evaluated on the same data for performance metrics

## Wyckoff Methodology Integration

The application integrates key Wyckoff concepts:

1. **Price Action Analysis**: The RL model learns to recognize favorable price patterns
2. **Volume Analysis**: Trading decisions consider volume as a confirming factor
3. **Wyckoff Phases**: The AI assistant helps identify accumulation, markup, distribution, and markdown phases
4. **Key Events**: Springs, upthrusts, and tests are explained by the AI assistant and potentially identified in the price data

## Project Structure

```
wyckoff-trading-assistant/
├── app.py                 # Main Flask application
├── trading_strategy.py    # RL trading algorithm
├── requirements.txt       # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css      # Application styling
│   └── js/
│       └── main.js        # Frontend JavaScript logic
├── templates/
│   ├── index.html         # Main application interface
│   ├── docs.html          # Documentation page
│   ├── 404.html           # Error page
│   └── 500.html           # Server error page
└── utils/
    ├── __init__.py
    ├── data_processing.py # Data handling functions
    └── model_handler.py   # Wyckoff model utilities
```

## Usage

### Dashboard
1. Enter a stock symbol in the search bar and click "Load"
2. View the stock price chart with technical indicators
3. Click "Run Backtest" to test the RL trading strategy

### Wyckoff Assistant
1. Navigate to the Wyckoff Assistant tab
2. Type your question about Wyckoff methodology or trading principles
3. View the AI-generated response based on Wyckoff principles

### Backtest Settings
1. Configure custom parameters (stock symbol, date range, initial capital)
2. Run the backtest with your settings
3. View detailed performance metrics and trading signals

## Future Enhancements

- Portfolio backtesting with multiple assets
- Custom technical indicators and trading rules
- Advanced Wyckoff pattern recognition
- Real-time trading signals and alerts
- Enhanced visualization of Wyckoff market phases
- User accounts to save preferences and backtest history

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The Wyckoff methodology originated from Richard D. Wyckoff's work in the early 20th century
- Q-learning algorithm developed by Watkins and Dayan (1992)
- Special thanks to the open-source communities behind Flask, PyTorch, and Chart.js

## Contact
- For questions or support, please contact:
- Abhishek Sagar Sanda - [sabhisheksagar200@gmail.com](sabhisheksagar200@gmail.com)
- Project Link: [SandaAbhishekSagar](https://github.com/SandaAbhishekSagar)
- [Portfolio](https://abhisheksagar.netlify.app)
---


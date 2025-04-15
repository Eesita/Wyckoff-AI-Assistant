from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import custom modules
from trading_strategy import StockTradingEnv, q_learning, discretize
from utils.data_processing import get_stock_data, format_data_for_chart, generate_basic_simulated_data, generate_simulated_data
from utils.model_handler import WyckoffModelHandler

app = Flask(__name__)

# Initialize Wyckoff model handler with direct path to .pth file
wyckoff_model = WyckoffModelHandler(model_path="/assets/transformer_chatbot_gpu_deco_2.pth",
                                   data_path="/assets/Cleaned_Wyckoff_QA_Dataset.csv" )

# Wyckoff Q&A endpoint
@app.route('/api/wyckoff_chat', methods=['POST'])
def wyckoff_chat():
    data = request.json
    question = data.get('message', '')
    
    logger.info(f"Received chat question: {question}")
    
    try:
        # Try to generate a response using the model
        response = wyckoff_model.generate_response(question)
        
        # If model failed or is not available, use fallback
        if not response:
            logger.info("Model response not available, using fallback")
            response = wyckoff_model.get_fallback_response(question)
            
        logger.info(f"Responding with: {response[:50]}...")
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in wyckoff_chat: {str(e)}")
        return jsonify({"response": "I apologize, but I'm having trouble processing your request right now."}), 500

# Stock data retrieval endpoint
@app.route('/api/stock_data', methods=['GET'])
def fetch_stock_data():
    symbol = request.args.get('symbol', 'NVDA')
    period = request.args.get('period', '1y')
    include_indicators = request.args.get('indicators', 'false').lower() == 'true'
    
    logger.info(f"Fetching stock data for {symbol}, period: {period}")
    
    try:
        # Get stock data using our utility function
        data = get_stock_data(symbol, period=period)
        
        if data is None or len(data) < 5:  # Require at least 5 data points
            logger.warning(f"Insufficient data for {symbol}, generating simulated data")
            data = generate_simulated_data(symbol)  # Pass the symbol to get appropriate simulated data
        
        # Format data for chart.js
        chart_data = format_data_for_chart(data, include_indicators)
        
        # Include the symbol in the response for verification
        chart_data['symbol'] = symbol
        return jsonify(chart_data)
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        # Return simulated data in case of error
        simulated_data = generate_simulated_data(symbol)  # Use the correct symbol
        chart_data = format_data_for_chart(simulated_data, include_indicators)
        chart_data['symbol'] = symbol
        return jsonify(chart_data)

# Backtest trading strategy endpoint
@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    data = request.json
    symbol = data.get('symbol', 'AAPL')
    start_date = data.get('start_date', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    initial_capital = float(data.get('initial_capital', 50000))
    
    logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
    
    try:
        # Get stock data using our utility function
        stocks = get_stock_data(symbol, start_date=start_date, end_date=end_date)
        
        if stocks is None or stocks.empty:
            return jsonify({"error": f"No data available for {symbol} in the specified date range"}), 400
            
        stock_prices = stocks['Close'].values
        
        # Initialize environment and run Q-learning
        env = StockTradingEnv(stock_prices)
        env.cash = initial_capital  # Set initial capital from request
        q_learning(env)
        
        # Test the learned policy
        state = env.reset()
        done = False
        portfolio_values = []
        actions_taken = []
        execution_prices = []
        dates = stocks.index.strftime('%Y-%m-%d').tolist()
        
        # Get global q_table from the trading_strategy module
        from trading_strategy import q_table
        
        if q_table is None:
            return jsonify({"error": "Q-learning failed to initialize the Q-table"}), 500
        
        while not done:
            price_bin = discretize(state[0], stock_prices)
            position = int(state[1])
            action = np.argmax(q_table[price_bin, position])
            next_state, reward, done = env.step(action)
            
            portfolio_values.append(float(env.portfolio_value))  # Convert numpy types to Python native types
            actions_taken.append(int(action))
            execution_prices.append(float(state[0]))
            
            state = next_state
        
        # Prepare results
        initial_value = initial_capital
        final_value = portfolio_values[-1] if portfolio_values else initial_value
        roi = ((final_value-initial_value) / initial_value ) * 100
        
        results = {
            'dates': dates[1:len(portfolio_values) + 1],  # Dates align with actions
            'portfolio_values': portfolio_values,
            'actions': actions_taken,
            'execution_prices': execution_prices,
            'action_labels': ['Hold', 'Buy', 'Sell'],
            'final_value': final_value,
            'initial_value': initial_value,
            'roi': roi,
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        }
        
        logger.info(f"Backtest completed. ROI: {roi:.2f}%, Final Value: ${final_value:.2f}")
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 400

# Main route
@app.route('/')
def index():
    return render_template('index.html')

# Documentation route
@app.route('/docs')
def docs():
    return render_template('docs.html')

# Error handler for 404
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Error handler for 500
@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# Initialize the application
def initialize_app():
    logger.info("Initializing application...")
    # Try to load the Wyckoff model
    success = wyckoff_model.load_model()
    if success:
        logger.info("Wyckoff model loaded successfully")
    else:
        logger.warning("Wyckoff model could not be loaded, will use fallback responses")

if __name__ == '__main__':
    # Initialize the app before running
    initialize_app()
    
    # Run the Flask application
    logger.info("Starting Flask application")
    app.run(debug=True, host='0.0.0.0', port=5000)
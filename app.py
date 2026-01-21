#!/usr/bin/env python3
"""
Web interface for UFC fight outcome prediction.
Simple Flask app to make predictions using trained models.
"""

from flask import Flask, render_template, request, jsonify
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict_fight import predict_fight
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Available models
MODELS = [
    'xgboost',
    'lightgbm',
    'logistic_regression',
    'random_forest',
    'gradient_boosting'
]

@app.route('/')
def index():
    """Render the main prediction page."""
    return render_template('index.html', models=MODELS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get form data
        fighter1 = request.form.get('fighter1', '').strip()
        fighter2 = request.form.get('fighter2', '').strip()
        fight_date = request.form.get('fight_date', '').strip()
        model_name = request.form.get('model', 'xgboost').strip()
        
        # Validate inputs
        if not fighter1 or not fighter2:
            return jsonify({
                'success': False,
                'error': 'Please provide both fighter names.'
            }), 400
        
        # Default to today's date if not provided
        if not fight_date:
            fight_date = datetime.now().strftime("%Y-%m-%d")
        
        # Validate date format
        try:
            datetime.strptime(fight_date, "%Y-%m-%d")
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid date format. Please use YYYY-MM-DD format.'
            }), 400
        
        # Validate model
        if model_name not in MODELS:
            model_name = 'xgboost'
        
        # Make prediction
        logger.info(f"Prediction request: {fighter1} vs {fighter2} on {fight_date} using {model_name}")
        result = predict_fight(fighter1, fighter2, fight_date, model_name)
        
        # Check for errors
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
        # Format response
        response = {
            'success': True,
            'fighter1': result['fighter1'],
            'fighter2': result['fighter2'],
            'fight_date': result['fight_date'],
            'predicted_winner': result['predicted_winner'],
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'fighter1_win_probability': result['fighter1_win_probability'],
            'fighter2_win_probability': result['fighter2_win_probability'],
            'model_used': result['model_used']
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)


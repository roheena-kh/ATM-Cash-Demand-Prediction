from flask import Blueprint, render_template, request, jsonify
import joblib
import os

daily_bp = Blueprint('daily', __name__)

# Load models and ATM lists for both currencies
currency_map = {
    'AFN': {
        'model_path': 'atm_cash_forecast_model.pkl',
        'data_path': os.path.join('data', 'base_data.csv'),
        'model': None,
        'atm_list': []
    },
    'USD': {
        'model_path': 'usd_atm_cash_forecast_model.pkl',
        'data_path': os.path.join('data', 'base_data.csv'),
        'model': None,
        'atm_list': []
    }
}

# Load models on blueprint init
for currency in currency_map:
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), currency_map[currency]['model_path'])
        model = joblib.load(model_path)
        atm_features = [f.split('_')[1] for f in model.get_booster().feature_names if f.startswith('TERM_')]
        currency_map[currency]['model'] = model
        currency_map[currency]['atm_list'] = sorted(list(set(atm_features)))
    except Exception as e:
        print(f"Error loading {currency} model: {str(e)}")

@daily_bp.route('/', methods=['GET'])
def daily_home():
    return render_template('daily.html', 
                         currencies=list(currency_map.keys()),
                         initial_atm_list=currency_map['AFN']['atm_list'])

@daily_bp.route('/get-atms', methods=['POST'])
def get_atms():
    currency = request.json.get('currency')
    if currency not in currency_map:
        return jsonify({'error': 'Invalid currency'}), 400
    return jsonify({'atm_list': currency_map[currency]['atm_list']})

@daily_bp.route('/predict', methods=['POST'])
def predict():
    try:
        currency = request.form.get('currency')
        
        if not currency or currency not in currency_map:
            return jsonify({'error': 'Invalid currency selection'}), 400
            
        model_data = currency_map[currency]

        if currency == 'AFN':
            from pipeline import predict_future
        else:
            from usd_pipeline import predict_future
            
        predictions = predict_future()
        predictions_list = []
        for _, row in predictions.iterrows():
            predictions_list.append({
                'atm_id': row['ATM_ID'],
                'prediction_date': str(row['NEXT_DATE']),
                'predicted_amount': float(row['PREDICTED_AMOUNT'])
            })
        
        return jsonify({
            'currency': currency,
            'predictions': predictions_list
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
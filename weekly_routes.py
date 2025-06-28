from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import joblib
import os

weekly_bp = Blueprint('weekly', __name__)

# Weekly model configuration
weekly_currency_map = {
    'AFN': {
        'model_path': 'atm_weekly_cash_forecast_model.pkl',
        'data_path': os.path.join('data', 'base_data.csv'),
        'model': None,
        'atm_list': []
    },
    'USD': {
        'model_path': 'atm_weekly_usd_cash_forecast_model.pkl',
        'data_path': os.path.join('data', 'usd_weekly_base_data.csv'),
        'model': None,
        'atm_list': []
    }
}

# Load weekly models
for currency in weekly_currency_map:
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), weekly_currency_map[currency]['model_path'])
        model_data = joblib.load(model_path)
        label_encoder = model_data['label_encoder']
        
        # Get path to current currency's data file
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), weekly_currency_map[currency]['data_path'])
        
        # Read data file and extract unique ATM IDs
        df = pd.read_csv(data_path)
        term_ids = df['TERM_ID'].unique().tolist() 
        
        atm_list = sorted(term_ids)
    
        weekly_currency_map[currency]['model'] = model_data['model']
        weekly_currency_map[currency]['atm_list'] = atm_list
        
    except Exception as e:
        print(f"Error loading weekly {currency} model: {str(e)}")

@weekly_bp.route('/', methods=['GET'])
def weekly_home():
    return render_template('weekly.html', 
                         currencies=list(weekly_currency_map.keys()),
                         initial_atm_list=weekly_currency_map['AFN']['atm_list'])

@weekly_bp.route('/get-atms', methods=['POST'])
def get_atms():
    currency = request.json.get('currency')
    if currency not in weekly_currency_map:
        return jsonify({'error': 'Invalid currency'}), 400
    return jsonify({'atm_list': weekly_currency_map[currency]['atm_list']})

@weekly_bp.route('/predict', methods=['POST'])
def predict():
    try:
        currency = request.form.get('currency')
        
        if not currency or currency not in weekly_currency_map:
            return jsonify({'error': 'Invalid currency selection'}), 400
            
        model_data = weekly_currency_map[currency]

        if currency == 'AFN':
            from weekly_afn_pipeline import predict_future
        else:
            from weekly_usd_pipeline import predict_future
            
        predictions = predict_future()
        predictions_list = []
        for _, row in predictions.iterrows():
            predictions_list.append({
                'atm_id': row['ATM_ID'],
                'prediction_date': str(row['NEXT_WEEK_START']),
                'predicted_amount': float(row['PREDICTED_AMOUNT'])
            })
        
        return jsonify({
            'currency': currency,
            'predictions': predictions_list
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta

usd_atms = ['ABI0034', 'ABI0040', 'ABI0043', 'ABI0051', 'ABI0090', 'ABI0101',
       'ABI0123', 'ABI0114', 'ABI0100', 'ABI0113', 'ABI0045', 'ABI0046']

# Load model from current directory
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usd_atm_cash_forecast_model.pkl')
model = joblib.load(MODEL_PATH)

def preprocess_input(df, atm_list):
    """Preprocess data for prediction"""
    try:
        df = df[df['TXN_CCY_CODE'] == "USD"].drop('TXN_CCY_CODE', axis=1)
        df = df[df['TRN_AMOUNT'] != 0]
        # Convert and sort dates
        df['TRN_DT'] = pd.to_datetime(df['TRN_DT'], format='%Y-%m-%d')
        df = df.sort_values(by='TRN_DT').reset_index(drop=True)
        
        # Group transactions by ATM and date
        grouped = df.groupby(['TERM_ID', 'TRN_DT'], as_index=False)['TRN_AMOUNT'].sum()
        grouped = grouped.sort_values('TRN_DT').reset_index(drop=True)
        
        # Create working copy for features
        processed = grouped.copy()
        
        # Date-based features
        processed['DAYOFWEEK'] = processed['TRN_DT'].dt.dayofweek
        processed['QUARTER'] = processed['TRN_DT'].dt.quarter
        processed['YEAR'] = processed['TRN_DT'].dt.year
        processed['MONTH'] = processed['TRN_DT'].dt.month
        processed['DAYOFYEAR'] = processed['TRN_DT'].dt.dayofyear
        processed['DAYOFMONTH'] = processed['TRN_DT'].dt.day
        processed['WORKDAY'] = processed['DAYOFWEEK'].apply(lambda x: 0 if x == 5 else 1)
        processed['SALARY'] = processed['DAYOFMONTH'].apply(lambda x: 0 if x in [1,2,3,4,5,27,28,29,30,31] else 1)
        
        # Lag features with groupby
        processed["lag_trn_1"] = processed.groupby("TERM_ID")['TRN_AMOUNT'].shift(1)
        processed["lag_trn_2"] = processed.groupby("TERM_ID")['TRN_AMOUNT'].shift(2)
        processed["lag_trn_3"] = processed.groupby("TERM_ID")['TRN_AMOUNT'].shift(3)
        processed["lag_trn_7"] = processed.groupby("TERM_ID")['TRN_AMOUNT'].shift(7)
        
        # Rolling means with minimum periods
        processed["mean_trn_3"] = processed.groupby("TERM_ID")['TRN_AMOUNT'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        processed["mean_trn_7"] = processed.groupby("TERM_ID")['TRN_AMOUNT'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        processed["mean_trn_15"] = processed.groupby("TERM_ID")['TRN_AMOUNT'].transform(
            lambda x: x.rolling(15, min_periods=1).mean()
        )
        
        # One-hot encoding for ATMs
        processed = pd.get_dummies(processed, columns=["TERM_ID"], prefix="TERM")
        
        # Ensure all expected ATM columns exist
        for atm in atm_list:
            col_name = f'TERM_{atm}'
            if col_name not in processed.columns:
                processed[col_name] = 0
                
        return processed, grouped

    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {str(e)}")

def predict_future():
    """Generate predictions for the next day"""
    try:
        # Load historical data
        DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'base_data.csv')
        df = pd.read_csv(DATA_PATH)
        
        atm_list = usd_atms  # predefined ATM list
        print("Predicting for these ATMs:", atm_list)
        
        # Preprocess data to get grouped transactions and features
        processed_data, original_grouped = preprocess_input(df, atm_list)
        
        # Generate new rows for next day predictions
        valid_atms = []
        new_rows = []
        for atm in atm_list:
            atm_data = original_grouped[original_grouped['TERM_ID'] == atm]
            if atm_data.empty:
                continue
            valid_atms.append(atm)
            
            last_row = atm_data.iloc[-1]
            last_date = last_row['TRN_DT']
            next_date = last_date + timedelta(days=1)
            
            # Date features for next day
            dayofweek = next_date.dayofweek
            quarter = next_date.quarter
            year = next_date.year
            month = next_date.month
            dayofyear = next_date.dayofyear
            dayofmonth = next_date.day
            workday = 0 if dayofweek == 5 else 1
            salary = 0 if dayofmonth in [1,2,3,4,5,27,28,29,30,31] else 1
            
            # Lag features from historical data
            lag_trn_1 = last_row['TRN_AMOUNT']
            
            lag_7_date = next_date - timedelta(days=7)
            lag_7_row = atm_data[atm_data['TRN_DT'] == lag_7_date]
            lag_trn_7 = lag_7_row['TRN_AMOUNT'].values[0] if not lag_7_row.empty else 0
            
            lag_2_date = next_date - timedelta(days=2)
            lag_2_row = atm_data[atm_data['TRN_DT'] == lag_2_date]
            lag_trn_2 = lag_2_row['TRN_AMOUNT'].values[0] if not lag_2_row.empty else 0

            lag_3_date = next_date - timedelta(days=3)
            lag_3_row = atm_data[atm_data['TRN_DT'] == lag_3_date]
            lag_trn_3 = lag_3_row['TRN_AMOUNT'].values[0] if not lag_3_row.empty else 0
            # Rolling means from historical data
            mean_trn_3 = atm_data['TRN_AMOUNT'].tail(3).mean()
            mean_trn_7 = atm_data['TRN_AMOUNT'].tail(7).mean()
            mean_trn_15 = atm_data['TRN_AMOUNT'].tail(15).mean()
            
            # Create new row with features
            new_row = {
                'DAYOFWEEK': dayofweek,
                'QUARTER': quarter,
                'YEAR': year,
                'MONTH': month,
                'DAYOFYEAR': dayofyear,
                'DAYOFMONTH': dayofmonth,
                'WORKDAY': workday,
                'SALARY': salary,
                'lag_trn_1': lag_trn_1,
                'lag_trn_2': lag_trn_2,
                'lag_trn_3': lag_trn_3,
                'lag_trn_7': lag_trn_7,
                'mean_trn_3': mean_trn_3,
                'mean_trn_7': mean_trn_7,
                'mean_trn_15': mean_trn_15
            }
            
            # Add one-hot encoded ATM columns
            for a in atm_list:
                new_row[f'TERM_{a}'] = 1 if a == atm else 0
            new_rows.append(new_row)
        
        if not new_rows:
            return pd.DataFrame(columns=['ATM_ID', 'NEXT_DATE', 'PREDICTED_AMOUNT'])
        
        # Prepare prediction data with correct feature order
        future_data = pd.DataFrame(new_rows)
        expected_features = model.get_booster().feature_names
        for feature in expected_features:
            if feature not in future_data.columns:
                future_data[feature] = 0
        future_data = future_data[expected_features]
        
        # Generate predictions
        predictions = model.predict(future_data)
        
        # Format output with ATM ID and next date
        output = []
        for atm, pred in zip(valid_atms, predictions):
            last_date = original_grouped[original_grouped['TERM_ID'] == atm]['TRN_DT'].max()
            next_date_str = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            output.append({
                'ATM_ID': atm,
                'NEXT_DATE': next_date_str,
                'PREDICTED_AMOUNT': int(round(pred, 0))
            })
        
        return pd.DataFrame(output)
    
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    predictions = predict_future()
    predictions.to_csv('usd_predictions.csv', index=False)
    print("USD Predictions generated successfully:")
    print(predictions.head())